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
    """ Input folder name, output sorted list of folders/paths inside it. """
    names_list=sorted(os.listdir(folder))
    paths_list=[os.path.join(folder, names_list[i]) for i in range(len(names_list))]
    return paths_list

def load_model(opt):
    """ Function to load the specified model. """

    model_name=opt.model
    
    n_classes=opt.n_classes    
    ###### add pretrained options
    if opt.no_rgb:
        n_channels=len(opt.channels)
    else:
        n_channels=len(opt.channels)+3

    if model_name == 'unet':
        model=smp.Unet(encoder_name='resnet50',
                       encoder_weights='imagenet',
                       in_channels=n_channels,
                       classes=n_classes)
    elif model_name == 'unet++':
        model=smp.UnetPlusPlus(encoder_name='resnet34',
                               encoder_weights='imagenet',
                               in_channels=n_channels,
                               classes=n_classes)
    
    elif model_name == 'deeplabv3':
        model=smp.DeepLabV3(encoder_name='resnet34',
                            encoder_weights='imagenet',
                            in_channels=n_channels,
                            classes=n_classes)
        
    elif model_name == 'deeplabv3+':
        model=smp.DeepLabV3Plus(encoder_name='resnet34',
                                encoder_weights='imagenet',
                                in_channels=n_channels,
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
    """ Function to load the chosen learning rate scheduler. """

    sched_name=opt.scheduler

    if sched_name=='cosine':
        scheduler=lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=opt.T_0, T_mult=2, eta_min=opt.eta_min)

    elif sched_name=='step':
        scheduler=lr_scheduler.StepLR(optimizer=optimizer, step_size=opt.step, gamma=opt.factor)
    
    elif sched_name=='triangular':
        scheduler=lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr,
                                        step_size_down=opt.step_size_down, step_size_up=opt.step_size_up, mode='triangular2')
    
    elif sched_name=='none':
        scheduler=None

    return scheduler

def load_optimizer(opt, model):
    """ Function to load the chosen optimizer (adam or sgd) """

    if opt.optimizer=='sgd':
        optimizer=optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)

    elif opt.optimizer=='adam':
        optimizer=optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    return optimizer

def load_loss(opt):
    """ Function to load the loss function. """

    if opt.loss=='crossentropy':
        loss=nn.CrossEntropyLoss()
    elif opt.loss=='jaccard':
        loss=smp.losses.JaccardLoss(mode='multiclass')
    

    return loss

def default_classes(opt):
    """ Function that returns default classes for the waste dataset.
        Returns a dictionary with classes and their segmentation value
    """

    if opt.n_classes==6:
        classes={'grass':0, 'obstacle':1, 'road':2, 'trash':3, 'vegetation':4, 'sky':5}
    elif opt.n_classes==5:
        classes={'grass':0, 'obstacle':1, 'road':2, 'trash':3, 'vegetation':4}
    return classes

def metric_names(opt):
    """ Function to get the name of each calculated metric. 
        Example: miou, iou_class1...
    """

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
    """ Function to initialize a file with the parameters used to train (parameters.csv) and a file where the metrics are going to be saved on (metrics.csv)"""

    parameters_file=os.path.join(opt.save_folder, 'parameters.csv')
    parameters={'channels':opt.channels, 
                'data_dir':opt.data_dir,
                'model':opt.model,
                'size':opt.size,
                'n_classes':opt.n_classes,
                'classes':opt.classes,
                'mean':opt.mean,
                'std':opt.std,
                'no_rgb':opt.no_rgb
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

    train_metric_file=os.path.join(opt.save_folder, 'train_metrics.csv')
    with open(train_metric_file, mode='w') as file:
        for metric in metrics:
            file.write(f'{metric},')
        file.write('\n')
    return parameters_file, metric_file, train_metric_file

def save_metrics(opt, valid_metrics, train_metrics):
    """ Function to save the metrics on metrics.csv for each epoch """

    with open(opt.metric_file, mode='a') as file:
        for metric in valid_metrics:
            _metric=round(metric.item()*100, 2)
            file.write(f'{_metric},')
        file.write('\n')
        
    with open(opt.train_metric_file, mode='a') as file:
        for metric in train_metrics:
            _metric=round(metric.item()*100, 2)
            file.write(f'{_metric},')
        file.write('\n')

def save_model(opt, model, model_name):
    """ Function to save the parameters of the last model (model.pth) """

    save_path=os.path.join(opt.save_folder, f'{model_name}.pth')
    torch.save(model.state_dict(), save_path)

def read_csv(opt):
    """ Function to read csv files into dictionaries """

    csv_path=os.path.join(opt.model_dir, 'parameters.csv')
    with open(csv_path, 'r') as file:
        reader=csv.DictReader(file)
        data=[row for row in reader][0]
    
    return data

def get_colormap(opt):
    """ Create a colormap to print the masks.

    """
    colors=['#00ff00', '#0033cc', '#a6a6a6', '#ffff00', '#18761c', '#18ffff']
    values=np.arange(opt.n_classes)
    value_to_color = {val: col for val, col in zip(values, colors)}
    cmap=ListedColormap([value_to_color[val] for val in values])

    return cmap

def get_folds(opt):
    """ Function to get a list for the training folds"""

    n_folds=len(os.listdir(opt.data_dir))
    train_folds=list(range(0,n_folds))
    opt.train_folds=[i for i in train_folds if i not in opt.valid_folds and i not in opt.test_folds]

def create_bash(opt):
    model=os.path.basename(opt.save_folder)
    data_folder=os.path.basename(opt.data_dir)
    test_file_folder=os.path.join('scripts', model)
    if not os.path.isdir(test_file_folder):
        os.makedirs(test_file_folder)
    test_file_path=os.path.join(test_file_folder, 'test_script.sh')
    with open(test_file_path, mode='w') as file:
        file.write('#!/bin/sh\n')
        file.write(f'python3 test.py --data_dir {data_folder} --k_fold --model_dir {model}')