from .base_options import BaseOptions
import utils.utils as utils

class TestOptions(BaseOptions):
    
    def initialize(self, parser):
        parser=BaseOptions.initialize(self, parser)

        self.isTrain=False



def get_test_options(opt):
    data=utils.read_csv(opt)
    opt.channels=data['channels']
    opt.model=data['model']
    opt.size=data['size']
    opt.n_classes=data['n_classes']
    opt.mean=data['mean']
    opt.std=data['std']