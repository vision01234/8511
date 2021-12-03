import numpy as np 
import torch 
from torch.autograd import Variable

def mixup_data(x, y, alpha=1.0, use_cuda=True): #y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x.cuda(), y_a.cuda(), y_b.cuda(), lam



def mixup(images, target):
    images, targets_a, targets_b, lam = mixup_data(images, target) #targets_a, targets_b, lam
    # images = map(Variable, (images)) # , targets_a, targets_b
    
    return images, targets_a, targets_b, lam



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
def cutmix(images, y):
    lam = np.random.beta(1, 1)
    rand_index = torch.randperm(images.size()[0]).cuda()
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    # compute output
    return images, target_a, target_b, lam


def puzzlemix(input):
    adv_p = 0.1
    adv_eps = 10.0
    adv_mask1 = np.random.binomial(n=1, p=adv_p)
    adv_mask2 = np.random.binomial(n=1, p=adv_p)

    mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]], dtype=torch.float32).reshape(1,3,1,1).cuda()
    std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]], dtype=torch.float32).reshape(1,3,1,1).cuda()

    # random start
    if (adv_mask1 == 1 or adv_mask2 == 1):
        noise = torch.zeros_like(input).uniform_(-adv_eps/255., adv_eps/255.)
        input_orig = input * std + mean
        input_noise = input_orig + noise
        input_noise = torch.clamp(input_noise, 0, 1)
        noise = input_noise - input_orig
        input_noise = (input_noise - mean)/std
        input = input_noise.cuda()
    else:
        input_var = Variable(input, requires_grad=True)
    target_var = Variable(target)


def ricap(images, targets):

    beta = 0.3 #self.beta  # hyperparameter

    # size of image
    I_x, I_y = images.size()[2:]

    # generate boundary position (w, h)
    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    # select four images
    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        index = torch.randperm(images.size(0)).cuda()
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = targets[index]
        W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

    # patch cropped images
    patched_images = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
            torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)

    targets = (c_, W_)
    return patched_images, c_, W_