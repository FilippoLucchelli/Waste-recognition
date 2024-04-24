from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """ Class that includes options for training """

    def initialize(self, parser):        

        parser=BaseOptions.initialize(self, parser)

        parser.add_argument('--pretrained', action='store_true', help='use pretrained weights')
        parser.add_argument('--save_folder', help='folder where to save models, plots, etc...')
        parser.add_argument('--channels', nargs='*', default=[], help='additional channels on top of rgb')
        parser.add_argument('--loss', default='jaccard', choices=['jaccard', 'crossentropy'])
        parser.add_argument('--valid_folds', type=int, nargs='+', help='folds to use for validation')
        parser.add_argument('--test_folds', type=int, nargs='*', default=[],help='folds to use for testing')

        parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--scheduler', default='cosine', choices=['none', 'step', 'plateau', 'triangular'])
        parser.add_argument('--T_0', type=int, default=100, help='T_0 parameter for cosine scheduler')
        parser.add_argument('--eta_min', type=float, default=0.00001, help='minimum learning rate for cosine scheduler')
        parser.add_argument('--step', type=int, help='step size for step scheduler')
        parser.add_argument('--factor', type=float, help='multiplication factor for step or plateau schedulers')
        parser.add_argument('--patience', type=int, help='Number of epochs with no improvement after which learning rate will be reduced for cosine scheduler')
        parser.add_argument('--base_lr', type=float, help='base learning rate for triangular scheduler')
        parser.add_argument('--max_lr', type=float, help='max lr for triangular scheduler')
        parser.add_argument('--step_size_up', type=int, help='step size up for triangular scheduler')
        parser.add_argument('--step_size_down', type=int, help='step size down for triangular scheduler. If non specified, equal to size up')

        parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
        parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay parameter for sgd')
        parser.add_argument('--momentum', default=0.8, type=float)
        parser.add_argument('--epochs', type=int, default=700, help='number of epochs for training')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for training and validation')
        parser.add_argument('--transforms', nargs='*', default=['h_flip', 'crop'], help='transforms for training: [v_flip, h_flip, crop]')
        parser.add_argument('--probability', type=float, default=0.5, help='probablity of flips')
        parser.add_argument('--scale', type=float, default=0.5, help='min scale parameter in crop')

        self.isTrain=True


        return parser

