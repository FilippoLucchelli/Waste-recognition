import torch
from torchvision import transforms
import numpy as np
from utils.metrics import Metrics
import utils.utils as utils
import time

def train_epoch(model, trainloader, optimizer, criterion, device, opt):

    model.train()
    
    counter=0

    metrics={metric_name:0 for metric_name in utils.metric_names(opt)}  
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
        _metrics=mt.get_metrics(opt)
        for metric_name in _metrics:
            metrics[metric_name]+=_metrics[metric_name]                          
        
        counter+=1
    for metric_name in metrics:
         metrics[metric_name]/=counter

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
    tr=transforms.Compose([transforms.Resize((opt.size, opt.size)),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomVerticalFlip()])
    
    return tr
    