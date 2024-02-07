import torch
from utils.metrics import Metrics
import visdom
import utils.utils as utils
import numpy as np
import ast
import time

def test(opt, model, test_loader, device):
    model.eval()
    vis=visdom.Visdom()
    cmap=utils.get_colormap(opt)

    counter=0
    metrics={metric_name:0 for metric_name in utils.metric_names(opt)}
    with torch.no_grad():
        for img, mask in test_loader:        
            img,mask=img.to(device), mask.to(device)
            out=model(img)

            out_mask=cmap(torch.argmax(out, dim=1).squeeze().cpu().numpy()).transpose((2,0,1))
            _mask=cmap(mask.squeeze().cpu().numpy()).transpose((2,0,1))
            filler=np.zeros((4, 640, 10))
            img_print=np.concatenate([_mask, filler, out_mask], axis=2)
            vis.image(img_print, win='image', opts=dict(store_history=True))        

            
            mt=Metrics(out, mask)
            _metrics=mt.get_metrics(opt)
            for metric_name in _metrics:
                metrics[metric_name]+=_metrics[metric_name]                  
                       

            counter+=1
        for metric_name in metrics:
            metrics[metric_name]/=counter

    return metrics



def get_test_options(opt):
    data=utils.read_csv(opt)
    opt.channels=ast.literal_eval(data['channels'])
    opt.model=(data['model'])
    opt.size=ast.literal_eval(data['size'])
    opt.n_classes=ast.literal_eval(data['n_classes'])
    opt.mean=ast.literal_eval(data['mean'])
    opt.std=ast.literal_eval(data['std'])
    opt.pretrained=True