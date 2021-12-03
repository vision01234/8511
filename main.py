# Task Transfer Learning
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import os

from opts import parse_opts
from model import StudentModel, SmartModel
from function import train, validation
from utils import save_path, Logger
from dataload import dataLoadFunc


import warnings

warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    opt = parse_opts()
    
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    
    if opt.dataset == 'imagenet':
        torch.manual_seed(1000)
    if opt.student_model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.lr = 0.01
    
    train_loader, valid_loader = dataLoadFunc(opt)

    StudentModel = StudentModel(opt)
    StudentModel = StudentModel.to(device)
    smartModel = None
    
    if opt.smart_model:
        opt.is_smart_model = True
        smartModel = SmartModel(opt)
        smartModel = smartModel.to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        StudentModel = nn.DataParallel(StudentModel)
        parms = list(StudentModel.module.parameters())
    
        if opt.is_smart_model:
            smartModel = nn.DataParallel(smartModel)
     
    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        parms = list(StudentModel.parameters())

    optimizer = torch.optim.SGD(parms, opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        
    milestones = [int(i) for i in (opt.lr_decay_epochs).split(',')]

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # record training process
    savePath, date_method, save_model_path  = save_path(opt)
    train_logger = Logger(os.path.join(savePath, date_method, 'train.log'),
                                    ['epoch', 'loss', 'acc', 'lr'])
    val_logger = Logger(os.path.join(savePath, date_method, 'val.log'),
                                    ['epoch', 'loss', 'acc', 'best_acc', 'lr'])

    writer = SummaryWriter(os.path.join(savePath, date_method,'logfile'))
    # start training
    best_acc = 0
    for epoch in range(1, opt.epochs + 1):
        # train, test model
        train_losses, train_scores, = train([StudentModel, smartModel], device, train_loader, optimizer, epoch, opt)
        test_losses, test_scores = validation([StudentModel, smartModel], device, optimizer, valid_loader, opt)
        scheduler.step()
                
        # plot average of each epoch loss value
        train_logger.log({
                        'epoch': epoch,
                        'loss': train_losses.avg,
                        'acc': train_scores.avg,
                        'lr': optimizer.param_groups[0]['lr']
                    })
        if best_acc < test_scores.avg:
            best_acc = test_scores.avg
            torch.save({'state_dict': StudentModel.state_dict()}, os.path.join(save_model_path, 'student_best.pth'))
        val_logger.log({
                        'epoch': epoch,
                        'loss': test_losses.avg,
                        'acc': test_scores.avg,
                        'best_acc' : best_acc,
                        'lr': optimizer.param_groups[0]['lr']
                    })
        writer.add_scalar('Loss/train', train_losses.avg, epoch)
        writer.add_scalar('Loss/test', test_losses.avg, epoch)
        
        writer.add_scalar('scores/train', train_scores.avg, epoch)
        writer.add_scalar('scores/test', test_scores.avg, epoch)
        torch.save({'state_dict': StudentModel.state_dict()}, os.path.join(save_model_path, 'student_lastest.pth'))  # save spatial_encoder
       
