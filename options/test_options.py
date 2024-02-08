from .base_options import BaseOptions
import utils.utils as utils

class TestOptions(BaseOptions):
    """ Class that includes options for testing """
    
    def initialize(self, parser):
        parser=BaseOptions.initialize(self, parser)

        parser.add_argument('--test_folds', type=int, nargs='*', help='folds to use for testing')

        self.isTrain=False
        return parser

