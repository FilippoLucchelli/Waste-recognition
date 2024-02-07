import os
import csv
import numpy as np
from models.MSNet import MSNet
from models.ACNet import ACNet
import segmentation_models_pytorch as smp 
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap


def list_and_sort_paths(folder):
    names_list=sorted(os.listdir(folder))
    paths_list=[os.path.join(folder, names_list[i]) for i in range(len(names_list))]
    return paths_list

def load_model(opt):
    model_name=opt.model
    
    n_classes=opt.n_classes

    
    ###### add pretrained options 
        
    n_channels=len(opt.channels)+3


    if model_name == 'unet':
        model=smp.Unet(encoder_name='resnet34',
                       encoder_weights='imagenet',
                       in_channels=3,
                       classes=n_classes)
    elif model_name == 'unet++':
        model=smp.UnetPlusPlus(encoder_name='resnet34',
                               encoder_weights='imagenet',
                               in_channels=3,
                               classes=n_classes)
    
    elif model_name == 'deeplabv3':
        model=smp.DeepLabV3(encoder_name='resnet34',
                            encoder_weights='imagenet',
                            in_channels=3,
                            classes=n_classes)
        
    elif model_name == 'deeplabv3+':
        model=smp.DeepLabV3Plus(encoder_name='resnet34',
                                encoder_weights='imagenet',
                                in_channels=3,
                                classes=n_classes)
        
    elif model_name == 'acnet':
        model=ACNet(num_class=n_classes, pretrained=True)

    elif model_name=='msnet':
        model=MSNet(num_classes=n_classes, n_channels=n_channels)
    
    else:
        print('no such model')
        model=None
    
    if opt.pretrained or not opt.isTrain:        
        model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'model.pth')))


    return model

def load_scheduler(opt, optimizer):
    sched_name=opt.scheduler

    if sched_name=='cosine':
        scheduler=lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=opt.T_0, T_mult=2, eta_min=opt.eta_min)

    elif sched_name=='step':
        scheduler=lr_scheduler.StepLR(optimizer=optimizer, step_size=opt.step, gamma=opt.factor)
    
    elif sched_name=='triangular':
        scheduler=lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr,
                                        step_size_down=opt.step_size_down, step_size_up=opt.step_size_up, mode='triangular2')
        
    return scheduler

def load_optimizer(opt, model):
    if opt.optimizer=='sgd':
        optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)

    elif opt.optimizer=='adam':
        optimizer=optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    return optimizer

def load_loss(opt):
    if opt.loss=='crossentropy':
        loss=nn.CrossEntropyLoss()
    elif opt.loss=='jaccard':
        loss=smp.losses.JaccardLoss(mode='multiclass')
    

    return loss

def default_classes(opt):
    if opt.n_classes==6:
        classes={'grass':0, 'obstacle':1, 'road':2, 'trash':3, 'vegetation':4, 'sky':5}
    elif opt.n_classes==5:
        classes={'grass':0, 'obstacle':1, 'road':2, 'trash':3, 'vegetation':4}
    return classes

def metric_names(opt):
    class_names=opt.classes
    metrics=[]
    if 'iou' in opt.metrics:
        metrics.append('miou')
    if 'dice' in opt.metrics:
        metrics.append('mdice')
    for class_name in class_names:
        if 'iou' in opt.metrics:
            metrics.append(f'{class_name}_iou')
        if 'dice' in opt.metrics:
            metrics.append(f'{class_name}_dice')
    return metrics

def init_files(opt):
    parameters_file=os.path.join(opt.save_folder, 'parameters.csv')
    parameters={'channels':opt.channels, 
                'model':opt.model,
                'k_fold':opt.k_fold,
                'size':opt.size,
                'n_classes':opt.n_classes,
                'classes':opt.classes,
                'mean':opt.mean,
                'std':opt.std,
                'test_folds': opt.test_folds
                }
    with open(parameters_file, mode='w') as file:
        writer=csv.DictWriter(file, parameters.keys())
        writer.writeheader()
        writer.writerow(parameters)
    
    metric_file=os.path.join(opt.save_folder, 'metrics.csv')
    metrics=metric_names(opt)
    with open(metric_file, mode='w') as file:
        for metric in metrics:
            file.write(f'{metric},')
        file.write('\n')
    return parameters_file, metric_file

def save_metrics(opt, metrics):
    with open(opt.metric_file, mode='a') as file:
        for metric in metrics:
            _metric=round(metric.item()*100, 2)
            file.write(f'{_metric},')
        file.write('\n')

def save_model(opt, model):
    save_path=os.path.join(opt.save_folder, 'model.pth')
    torch.save(model.state_dict(), save_path)

def read_csv(opt):
    csv_path=os.path.join(opt.model_dir, 'parameters.csv')
    with open(csv_path, 'r') as file:
        reader=csv.DictReader(file)
        data=[row for row in reader][0]
    
    return data

def get_colormap(opt):
    """ Create a colormap to print the masks.

    args:
        n_classes (int): number of classes for the model

    """
    colors=['#00ff00', '#0033cc', '#a6a6a6', '#ffff00', '#18761c', '#18ffff']
    values=np.arange(opt.n_classes)
    value_to_color = {val: col for val, col in zip(values, colors)}
    cmap=ListedColormap([value_to_color[val] for val in values])

    return cmap

def get_folds(opt):
    n_folds=len(os.listdir(opt.data_dir))
    train_folds=list(range(0,n_folds))
    opt.train_folds=[i for i in train_folds if i not in opt.valid_folds and i not in opt.test_folds]
