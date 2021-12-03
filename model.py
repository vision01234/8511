import os
import torch
import torch.nn as nn


from models import model_dict
import torchvision
def StudentModel(opt):
    model = model_dict[opt.student_model](num_classes=opt.n_class)
    return model
def SmartModel(opt):
    model = model_dict[opt.smart_model](num_classes=opt.n_class)
    
    checkpoint = torch.load(os.path.join(opt.smart_model_path, 'student_lastest.pth'))#, map_location=lambda storage, loc: storage)

    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)


    return model
