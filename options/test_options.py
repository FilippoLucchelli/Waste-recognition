from .base_options import BaseOptions
import utils.utils as utils

class TestOptions(BaseOptions):
    """ Class that includes options for testing """
    
    def initialize(self, parser):
        parser=BaseOptions.initialize(self, parser)

        parser.add_argument('--test_folds', type=int, nargs='*', help='folds to use for testing')
        parser.add_argument('--print_images', action='store_true', help='print the results on visdom server')

        self.isTrain=False
        return parser

