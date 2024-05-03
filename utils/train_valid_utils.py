import torch
from torchvision import transforms
import numpy as np
from utils.metrics import Metrics
import utils.utils as utils
import time
import ast

def train_epoch(model, trainloader, optimizer, criterion, device, opt):

    model.train()
    
    counter=0

    metrics={metric_name:0 for metric_name in utils.metric_names(opt)}  #initialize metric dictionary
    tot_loss=0  
    
    for img, mask in trainloader:
        img, mask= img.to(device), mask.to(device)
        optimizer.zero_grad()
        output=model(img)
        loss=criterion(output, mask)
        loss.backward()
        optimizer.step()

        tot_loss+=loss

        mt=Metrics(output, mask)
        _metrics=mt.get_metrics(opt) #compute metrics for the batch
        for metric_name in _metrics:
            metrics[metric_name]+=_metrics[metric_name] #accumulate metrics                    
        
        counter+=1
    for metric_name in metrics:
         metrics[metric_name]/=counter #average metrics

    return metrics, tot_loss/counter

def valid_epoch(model, validloader, criterion, device, opt):
    model.eval()    
    
    with torch.no_grad():
        metrics={metric_name:0 for metric_name in utils.metric_names(opt)}   
        tot_loss=0
        counter=0

        for img, mask in validloader:
            img, mask = img.to(device), mask.to(device)
            output=model(img)
            loss=criterion(output, mask)
            
            mt=Metrics(output, mask)
            _metrics=mt.get_metrics(opt)
            for metric_name in _metrics:
                metrics[metric_name]+=_metrics[metric_name]                          
            
            counter+=1
        for metric_name in metrics:
            metrics[metric_name]/=counter

    return metrics, tot_loss/counter 

def get_transforms(opt):
    """ Function to get the chosen transforms """
    tr=[transforms.Resize(opt.size)]
    if 'v_flip' in opt.transforms:
        tr.append(transforms.RandomVerticalFlip(opt.probability))
    if 'h_flip' in opt.transforms:
        tr.append(transforms.RandomHorizontalFlip(opt.probability))
    if 'crop' in opt.transforms:
        tr.append(transforms.RandomResizedCrop(size=opt.size, scale=(opt.scale, 1.0)))
    
    transform=transforms.Compose(tr)
    return transform
    
def get_pretrained_options(opt):
    """ Function to load options from the trained model. """

    data=utils.read_csv(opt)
    opt.channels=ast.literal_eval(data['channels'])
    opt.model=(data['model'])
    opt.size=ast.literal_eval(data['size'])
    opt.n_classes=ast.literal_eval(data['n_classes'])
    opt.no_rgb=ast.literal_eval(data['no_rgb'])
