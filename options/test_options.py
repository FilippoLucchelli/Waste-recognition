from .base_options import BaseOptions
import utils.utils as utils

class TestOptions(BaseOptions):
    
    def initialize(self, parser):
        parser=BaseOptions.initialize(self, parser)

        self.isTrain=False
        return parser

