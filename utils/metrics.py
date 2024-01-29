import torch

def IoU(pred, target, n_class):

    pred_binary=pred==n_class
    target_binary=target==n_class

    if torch.sum(pred_binary)==0 and torch.sum(target_binary)==0:
        return None

    else:
        intersection=torch.logical_and(pred_binary, target_binary)
        union=torch.logical_or(pred_binary, target_binary)

