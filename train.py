import os
import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from custom_dataset import CustomDataset
from utils import utils
from utils.train_valid_utils import train_epoch, valid_epoch
import utils.visdom_utils as visdom_utils
import time


if __name__=='__main__':
    opt=TrainOptions().parse() #parse options
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    utils.get_folds(opt) #get folds for training

    train_set=CustomDataset(opt, opt.train_folds)
    opt.mean, opt.std=train_set.get_mean_std()
    train_set=CustomDataset(opt, opt.train_folds)
    valid_set=CustomDataset(opt, opt.valid_folds)

    opt.parameters_file, opt.metric_file=utils.init_files(opt)

    train_loader=DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_loader=DataLoader(valid_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    model=utils.load_model(opt).to(device)
    criterion=utils.load_loss(opt)
    optimizer=utils.load_optimizer(opt, model)
    scheduler=utils.load_scheduler(opt, optimizer=optimizer)
    
    plotters={}
    for metric_name in utils.metric_names(opt):
        plotters[metric_name]=visdom_utils.VisUtils(metric_name)

    for epoch in range(opt.epochs):

        train_metrics, train_loss=train_epoch(model=model, criterion=criterion,
                                              trainloader=train_loader,
                                              optimizer=optimizer, device=device,
                                              opt=opt)
        
        valid_metrics, valid_loss=valid_epoch(model=model, validloader=valid_loader,
                                              criterion=criterion, device=device, opt=opt)

        for metric_name in utils.metric_names(opt):
            plotters[metric_name].update_plot(epoch, [valid_metrics[metric_name].cpu(), train_metrics[metric_name].cpu()])
        
        utils.save_metrics(opt, valid_metrics.values())

        if scheduler is not None:
            scheduler.step()
            
    utils.save_model(opt, model)
