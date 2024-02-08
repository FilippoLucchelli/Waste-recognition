import argparse
import os
import utils.utils as utils
from utils.test_utils import get_test_options


class BaseOptions():
    """ This class defines options used during both training and test time. """
    
    def __init__(self):
        self.initialized=False

    def initialize(self, parser):
        
        parser.add_argument('--data_dir', required=True, help='path to images')
        parser.add_argument('--k_fold', action='store_true', help='use a k_fold structure')
        parser.add_argument('--model_dir', help='with saved model and csv files')

        parser.add_argument('--model', default='msnet', choices=['msnet', 'acnet', 'unet', 'unet++', 'deeplabv3', 'deeplabv3+'])
        parser.add_argument('--size', type=int, default=640, help='size of resized data in custom dataset class')
        parser.add_argument('--n_classes', type=int, default=6, help='number of classes for segmentation')

        parser.add_argument('--classes', nargs='+', help='name of classes')
        parser.add_argument('--metrics', nargs='+', default=['iou'], help='evaluation metrics')
        

        self.initialized=True
        return parser
        
    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser=parser
        return parser.parse_args()
    
    def parse(self):        
        
        self.opt=self.gather_options()
        
        if self.opt.classes is None:
            self.opt.classes=utils.default_classes(self.opt)

        if self.opt.model_dir is not None:
            self.opt.model_dir=os.path.join('results', self.opt.model_dir)

        if self.isTrain:
            self.opt.save_folder=os.path.join('results', self.opt.save_folder)
            if self.opt.save_folder is not None:
                os.makedirs(self.opt.save_folder, exist_ok=True)
        
        self.opt.isTrain=self.isTrain
        if not self.isTrain:
            get_test_options(self.opt)

        return self.opt
        