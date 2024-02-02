import torch
import utils.utils as utils


class Metrics:

    def __init__(self, pred, target):
        _pred=pred.clone()
        if len(pred.shape)!=len(target.shape):
            _pred=torch.argmax(_pred, dim=1)
        
        self.pred=_pred
        self.target=target

    def IoU(self, n_class):

        
        pred_binary=self.pred==n_class
        target_binary=self.target==n_class

        if torch.sum(pred_binary)==0 and torch.sum(target_binary)==0:
            return 0

        else:
            intersection=torch.logical_and(pred_binary, target_binary)
            union=torch.logical_or(pred_binary, target_binary)
            return torch.sum(intersection)/torch.sum(union)

    def dice(self, n_class):

        pred_binary=self.pred==n_class
        target_binary=self.target==n_class

        if torch.sum(pred_binary)==0 and torch.sum(target_binary)==0:
            return 0
        
        else:
            intersection=torch.logical_and(pred_binary, target_binary)
            summ=torch.sum(pred_binary)+torch.sum(target_binary)
            return (torch.sum(intersection)/torch.sum(summ))*2
        
    def accuracy(self):
        sum_correct=torch.sum(self.pred==self.target)
        tot_pixels=self.target.size(0)*self.target.size(1)
        return sum_correct/tot_pixels
    
    def get_metrics(self, opt):
        class_values=utils.default_classes(opt)
        metrics={}
        miou=0
        mdice=0
        
        if 'accuracy' in opt.metrics:
            metrics['accuracy']=self.accuracy()

        for class_name in class_values:
            if 'iou' in opt.metrics:
                metrics[f'{class_name}_iou']=self.IoU(class_values[class_name])
                miou+=metrics[f'{class_name}_iou']

            if 'dice' in opt.metrics:
                metrics[f'{class_name}_dice']=self.dice(class_values[class_name])
                mdice+=metrics[f'{class_name}_dice']

        if 'iou' in opt.metrics:
            metrics['miou']=miou/opt.n_classes
        
        if 'dice' in opt.metrics:
            metrics['mdice']=mdice/opt.n_classes

        return metrics

