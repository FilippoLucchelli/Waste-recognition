import torch
import numpy as np
from utils.metrics import Metrics

def train_epoch(model, trainloader, optimizer, criterion, device, n_classes):

    model.train()
    
    counter=0
    acc=0
    miou=0
    mdice=0
    iou={i:0 for i in range(n_classes)}
    dice={i:0 for i in range(n_classes)}
    
    for img, mask in trainloader:
        img, mask= img.to(device), mask.to(device)
        output=model(img)
        loss=criterion(output, mask)
        loss.backward()
        optimizer.step()

        metrics=Metrics(output, mask, n_classes).to(device)
        _miou, _mdice, _iou, _dice, _acc=metrics.get_metrics()
        miou+=_miou
        mdice+=_mdice
        acc+=_acc
        for i in range(n_classes):
            iou[i]+=_iou[i]
            dice[i]+=_dice[i]

        counter+=1
    for i in range(n_classes):
            iou[i]/=counter
            dice[i]/=counter
    return miou/counter, mdice/counter, iou, dice, acc

def valid_epoch(model, trainloader, criterion, device, n_classes):
    model.eval()

    counter=0
    acc=0
    miou=0
    mdice=0
    iou={i:0 for i in range(n_classes)}
    dice={i:0 for i in range(n_classes)}
    
    with torch.no_grad():
        for img, mask in trainloader:
            img, mask = img.to(device), mask.to(device)
            output=model(img)
            loss=criterion(img, mask)
            metrics=Metrics(output, mask, n_classes).to(device)
            _miou, _mdice, _iou, _dice, _acc=metrics.get_metrics()
            miou+=_miou
            mdice+=_mdice
            acc+=_acc
            for i in range(n_classes):
                iou[i]+=_iou[i]
                dice[i]+=_dice[i]

            counter+=1
        for i in range(n_classes):
                iou[i]/=counter
                dice[i]/=counter
    return miou/counter, mdice/counter, iou, dice, acc