import os
import torch
import numpy as np 
import utils.utils as utils

import torch 
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomDataset(Dataset):

    def __init__(self, data_dir, k_fold=True, fold_list=None, bands=[]):
        
        rgb_names=['red', 'green', 'blue']
        band_names=rgb_names+bands+['masks']

        if k_fold:
            folds=utils.list_and_sort_paths(data_dir)
            band_paths={}
            for n_fold, fold in enumerate(folds):
                if n_fold in fold_list:
                    for band_name in band_names:
                        if band_name not in band_paths:
                            band_paths[band_name]=[]
                        band_paths[band_name].extend(utils.list_and_sort_paths(os.path.join(fold, band_name)))
            self.band_paths=band_paths
            self.band_names=band_names
            self.bands=bands
    
    def __len__(self):
        return len(self.band_paths[self.band_names[0]])
    
    def __getitem__(self, index):
        
        data=[]
        for band_name in self.band_names:
            band=np.load(self.band_paths[band_name][index])     
            data.append(band)
        
        data_np=np.stack(data)
        data_torch=torch.from_numpy(data_np)
        
        img=data_torch[:-1]
        mask=data_torch[-1]

        return img, mask