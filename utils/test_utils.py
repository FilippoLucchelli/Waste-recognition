import torch
from utils.metrics import Metrics
import visdom
import utils.utils as utils
import numpy as np
import ast
import time


#### Utils for the test phase

def test(opt, model, test_loader, device):
    """ Function to test a trained model on a testset. Takes parameters from the training stage """

    model.eval()
    vis=visdom.Visdom()
    vis.close()
    cmap=utils.get_colormap(opt)

    counter=0
    metrics={metric_name:0 for metric_name in utils.metric_names(opt)} #initialize metric dict
    with torch.no_grad():
        for img, mask in test_loader:        
            img,mask=img.to(device), mask.to(device)
            out=model(img)

            #Move tensor to cpu and numpy, transform logits into number with argmax, then apply colormap and transpose to feed into visdom
            out_mask=cmap(torch.argmax(out, dim=1).cpu().numpy()).transpose((0,3,1,2)) 
            _mask=cmap(mask.cpu().numpy()).transpose((0,3,1,2)) 
            filler=np.zeros((_mask.shape[0], _mask.shape[1], _mask.shape[2], 10)) # White space between images
            
            imgs_print=np.concatenate([_mask, filler, out_mask], axis=3)

            for img_print in imgs_print:
                vis.image(img_print, win='image', opts=dict(store_history=True)) # Send to visdom server    

            
            mt=Metrics(out, mask)
            _metrics=mt.get_metrics(opt)
            for metric_name in _metrics:
                metrics[metric_name]+=_metrics[metric_name]                  
                       

            counter+=1
        for metric_name in metrics:
            metrics[metric_name]/=counter
        print_metrics(vis, metrics)

    return metrics



def get_test_options(opt):
    """ Function to load options from the training stage. """

    data=utils.read_csv(opt)
    opt.channels=ast.literal_eval(data['channels'])
    opt.model=(data['model'])
    opt.size=ast.literal_eval(data['size'])
    opt.n_classes=ast.literal_eval(data['n_classes'])
    opt.mean=ast.literal_eval(data['mean'])
    opt.std=ast.literal_eval(data['std'])
    if opt.test_folds==None:
        opt.test_folds=ast.literal_eval(data['test_folds'])
    opt.pretrained=True

def print_metrics(vis, metrics):
    """ Print metrics on visdom server """
    
    txt='Metrics: <br>'
    for metric_name in metrics:
        metric_val=np.round((metrics[metric_name].cpu().numpy())*100, decimals=2)        
        txt+=f'{metric_name}: {metric_val} <br>'
    
    vis.text(txt, win='metrics')