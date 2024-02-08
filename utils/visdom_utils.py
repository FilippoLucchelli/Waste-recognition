import visdom
import numpy as np
import torch


class VisUtils:
    """ Class to initialize and plot line plots on visdom """
    def __init__(self, metric_name, vis):
        self.metric_name=metric_name
        self.vis=vis
        self.plot_option=dict(title=metric_name, xlabel='Epoch', ylabel=metric_name, legend=['Validation', 'Training'])
        self.plot=self.vis.line(X=np.zeros((1,)), Y=np.zeros((1,2)),
                                opts=self.plot_option)

    def update_plot(self, epoch, metric_value):
        self.vis.line(X=torch.ones((1,2))*epoch, Y=torch.Tensor(metric_value).unsqueeze(0),
                win=self.plot,
                update='append' if epoch > 0 else None,
                opts=self.plot_option)
        
        

