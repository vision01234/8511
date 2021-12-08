import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, get_ap_score, accuracy
from tools.augmentation import mixup, cutmix, ricap
from sklearn.preprocessing import OneHotEncoder

def CrossEntropy(predicted, target):
    return torch.mean(torch.sum(-nn.Softmax()(target) * torch.nn.LogSoftmax()(predicted), 1))

class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target): 
        pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad(): # true_dist = pred.data.clone() 
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls)) 
            for i in range(target.size(0)):
                true_dist[i][int(target[i].data)] += self.confidence
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def model_fit(pred, target,train_method, T=1):
    labelsmoothingcrossentropy = LabelSmoothingLoss(classes = int(pred.size(1)))
    if train_method == 'vanilla': 
        loss = F.cross_entropy(pred, target)
    elif train_method == 'kd':
        loss = T*T*nn.KLDivLoss()(F.log_softmax(pred / T, dim=1), F.softmax(target / T, dim=1))
    elif train_method == 'SL':
        loss =  CrossEntropy(pred, target)
    elif train_method == 'label_smoothing':
        loss = labelsmoothingcrossentropy(pred, target) 
    return loss

def loss_fn(output_st, output_sm, labels, opt):
    softmax = nn.Softmax()
    totalLoss = 0
    kd_loss = 0
    if opt.method == 'SL':
        totalLoss += model_fit(output_st, output_sm, 'SL', T=opt.T)
        return totalLoss
    elif opt.method == 'label_smoothing':
        totalLoss = model_fit(output_st, labels, 'label_smoothing', T=opt.T)
        return totalLoss
    else:
        Student_loss = model_fit(output_st, labels, 'vanilla', T=opt.T)
        if opt.method == 'kd':
            kd_loss = model_fit(output_st, output_sm, 'kd', T=opt.T)
        
        totalLoss += Student_loss + opt.alpha * (kd_loss)
        return totalLoss




def train(model, device, train_loader, optimizer, epoch, opt):
    studentModel, smartModel = model
    studentModel.train()
    if opt.is_smart_model:
        smartModel = smartModel.eval()

    losses = AverageMeter()
    Targetscores = AverageMeter()

    N_count = 0  
    for batch_idx, (images, y) in enumerate(train_loader):
        images, y = images.to(device), y.to(device)
        N_count+= images.size(0)
        if opt.aug_method == 'mixup':
            images, targets_a, targets_b, lam = mixup(images, y)
        elif opt.aug_method == 'cutmix':
            images, target_a, target_b, lam = cutmix(images, y)
        elif opt.aug_method == 'ricap':
            images, c_, W_  = ricap(images, y)
            

        optimizer.zero_grad()
        
        output_st  = studentModel(images) #feature_st

        if opt.is_smart_model:
            with torch.no_grad():
                output_sm  = smartModel(images) #feature_sm

            loss = loss_fn(output_st, output_sm, y, opt)
                
        else:
            criterion = nn.CrossEntropyLoss()
            if opt.aug_method == 'mixup':
                loss = lam * criterion(output_st, targets_a) + (1 - lam) * criterion(output_st, targets_b)
            elif opt.aug_method == 'cutmix':
                loss = criterion(output_st, target_a) * lam + criterion(output_st, target_b) * (1. - lam)
            elif opt.aug_method == 'ricap':
                loss = sum([W_[k] * criterion(output_st, c_[k]) for k in range(4)])
            else:
                loss = model_fit(output_st, y, opt.method, T=opt.T)
            
        losses.update(loss.item(), images.size()[0])

        y_pred = torch.max(output_st, 1)[1]  
        step_score = accuracy(output_st.data, y.data, topk=(1,))[0]
        Targetscores.update(step_score,images.size()[0])        
      
        loss.backward()
        optimizer.step()

        if (batch_idx) % 10 == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), losses.avg, Targetscores.avg))
            
   
    return losses, Targetscores


def validation(model, device, optimizer, test_loader, opt):
    studentModel, smartModel = model

    studentModel.eval()
    if opt.is_smart_model:
        smartModel = smartModel.eval()
      
    accs = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        for images, y in test_loader:

            images, y = images.to(device), y.to(device)
           
            output_st = studentModel(images)
         
            if opt.is_smart_model:
                with torch.no_grad():
                    output_sm = smartModel(images)
                 

                loss = loss_fn(output_st, output_sm, y, opt)

            else:
                loss = model_fit(output_st, y, opt.method, T=opt.T)
            losses.update(loss.item(), images.size()[0])                
            prec = accuracy(output_st.data, y.data, topk=(1,))[0]
            accs.update(prec.item(), images.size()[0])
   
        

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(test_loader.dataset), losses.avg, accs.avg))
  
    
    return losses, accs

