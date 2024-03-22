from custom_dataset import CustomDataset
from options.test_options import TestOptions
import utils.utils as utils
import utils.test_utils as test_utils

import torch
from torch.utils.data import DataLoader
import time


if __name__=='__main__':
    opt=TestOptions().parse() #parsing options
        
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_set=CustomDataset(opt, opt.test_folds)
    test_loader=DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True)

    model=utils.load_model(opt).to(device)

    metrics=test_utils.test(opt, model, test_loader, device)
    


