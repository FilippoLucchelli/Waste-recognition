import os
import torch
import numpy as np 
import utils.utils as utils
import yaml

import torch 
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import time


class CustomDataset(Dataset):

    def __init__(self, opt, fold_list=None):
        
        rgb_names=['red', 'green', 'blue']
        band_names=rgb_names+opt.channels+['masks']

        if opt.k_fold:
            folds=utils.list_and_sort_paths(opt.data_dir)
            band_paths={}
            for n_fold, fold in enumerate(folds):
                if n_fold in fold_list:
                    for band_name in band_names:
                        if band_name not in band_paths:
                            band_paths[band_name]=[]
                        band_paths[band_name].extend(utils.list_and_sort_paths(os.path.join(fold, band_name)))
            self.band_paths=band_paths
            self.band_names=band_names
            self.bands=opt.channels
            self.opt=opt
    
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
        if self.opt.mean is not None:
            norm=transforms.Normalize(self.opt.mean, self.opt.std)
            img=norm(img)
            

        return img, mask.long()
    
    def get_mean_std(self):
        mean={band_name: 0.0 for band_name in self.band_names if band_name!='masks'}
        std={band_name: 0.0 for band_name in self.band_names if band_name != 'masks'}
        counter=0
        for band_name in self.band_names:
            if band_name!='masks':
                for band_path in self.band_paths[band_name]:
                    band=np.load(band_path)
                    mean[band_name]+=np.mean(band)
                    std[band_name]+=np.std(band)
                    counter+=1
                mean[band_name]/=counter
                std[band_name]/=counter
                counter=0

        return list(mean.values()), list(std.values())


class CustomDatasetYaml:

    def __init__(self, opt, phase='train', transforms=None):
        
        with open(opt.data_dir, 'r') as yml:
            conf=yaml.safe_load(yml)
        
        if phase=='train':
            folder=os.path.join(conf['root_dir'], conf['train_dir'])
        elif phase=='val':
            folder=os.path.join(conf['root_dir'], conf['val_dir'])
        elif phase=='test':
            folder=os.path.join(conf['root_dir'], conf['test_dir'])
        
        if opt.no_rgb:
            rgb_names=[]
        else:
            rgb_names=['red', 'green', 'blue']
        
        if opt.ground_truth:
            band_names=rgb_names+opt.channels+['masks']
        
        else:
            band_names=rgb_names+opt.channels

        band_folders=[os.path.join(folder, band) for band in band_names]
        band_paths={band: [] for band in band_names}

        for band_folder, band_name in zip(band_folders, band_names):
            band_paths[band_name]=utils.list_and_sort_paths(band_folder)
        
        self.band_paths=band_paths
        self.band_names=band_names
        self.transforms=transforms
        self.opt=opt
    
    def __len__(self):
        return len(self.band_paths[self.band_names[0]])
    
    def __getitem__(self, index):
        data=[]
        for band in self.band_names:
            path=self.band_paths[band][index]
            band=np.load(path)
            data.append(band)
        data=np.stack(data)
        data_torch=torch.from_numpy(data)

        if self.transforms is not None:
            data_torch=self.transforms(data_torch)

        if data_torch.shape[1]!=self.opt.size and data_torch.shape[2]!=self.opt.size:
            rs=transforms.Resize((self.opt.size, self.opt.size), antialias=True)
            data_torch=rs(data_torch)

        if self.opt.ground_truth:
            img=data_torch[:-1]
            mask=data_torch[-1]
        else:
            img=data_torch
        
        
        if self.opt.single_class:
            mask=(mask==3).int()

        if self.opt.mean is not None:
            norm=transforms.Normalize(self.opt.mean, self.opt.std)
            img=norm(img)

        if self.opt.ground_truth:
            return img, mask.long()
        else:
            return img
    
    def get_mean_std(self):
        mean={band_name: 0.0 for band_name in self.band_names if band_name!='masks'}
        std={band_name: 0.0 for band_name in self.band_names if band_name != 'masks'}
        counter=0
        for band_name in self.band_names:
            if band_name!='masks':
                for band_path in self.band_paths[band_name]:
                    band=np.load(band_path)
                    mean[band_name]+=np.mean(band)
                    std[band_name]+=np.std(band)
                    counter+=1
                mean[band_name]/=counter
                std[band_name]/=counter
                counter=0

        return list(mean.values()), list(std.values())
