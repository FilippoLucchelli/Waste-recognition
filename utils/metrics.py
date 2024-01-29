import torch


class Metrics:

    def __init__(self, pred, target, n_classes):
        if len(pred.shape)!=len(target.shape):
            pred=torch.argmax(pred, dim=1)
        
        self.pred=pred
        self.target=target
        self.n_classes=n_classes

    def IoU(self, n_class):

        
        pred_binary=self.pred==n_class
        target_binary=self.target==n_class

        if torch.sum(pred_binary)==0 and torch.sum(target_binary)==0:
            return None

        else:
            intersection=torch.logical_and(pred_binary, target_binary)
            union=torch.logical_or(pred_binary, target_binary)
            return torch.sum(intersection)/torch.sum(union)

    def dice(self, n_class):

        pred_binary=self.pred==n_class
        target_binary=self.target==n_class

        if torch.sum(pred_binary)==0 and torch.sum(target_binary)==0:
            return None
        
        else:
            intersection=torch.logical_and(pred_binary, target_binary)
            summ=torch.sum(pred_binary)+torch.sum(target_binary)
            return (torch.sum(intersection)/torch.sum(summ))*2
        
    def accuracy(self):
        sum_correct=torch.sum(self.pred==self.target)
        tot_pixels=self.target.size(0)*self.target.size(1)
        return sum_correct/tot_pixels
    
    def get_metrics(self):
        iou, dice={i:None for i in range(self.n_classes)}, {i:None for i in range(self.n_classes)}
        miou=0
        mdice=0
        for n_class in range(self.n_classes):
            iou[n_class]=self.IoU(n_class)
            dice[n_class]=self.dice(n_class)
            miou+=iou[n_class]
            mdice+=dice[n_class]

        miou/=self.n_classes
        mdice/=self.n_classes        
        acc=self.accuracy()
        return miou, mdice, iou, dice, acc

