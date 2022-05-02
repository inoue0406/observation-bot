import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os

# Pytorch custom dataset for artificially generated fields
# The class assumes the data to be in h5 format

class ArtfieldDataset(data.Dataset):
    def __init__(self,csv_file,root_dir,mode,tdim_use=12,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the radar data.
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            mode: "value" or "velocity"
                   if "value" is specified, value is returned as future data
                   if "velocity" is specified, velocity is returned as future data
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df_fnames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tdim_use = tdim_use
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.df_fnames)
        
    def __getitem__(self, index):
        fnames = self.df_fnames.iloc[index].loc['fname']
        # read X
        h5_name_all = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fname'])
        h5file = h5py.File(h5_name_all,'r')
        rain_all = h5file['R'][()].astype(np.float32)
        rain_all = np.maximum(rain_all,0) # replace negative value with 0
        rain_all = rain_all[:,None,:,:] # add "channel" dimension as 1
        rain_X = rain_all[0:self.tdim_use,:,:,:] # use time 0 to tdim_use as X
        # read Y
        if self.mode == "velocity":
            vel = h5file['V'][()].astype(np.float32)
            vel = vel[None,:,:,:] # add "time" dimension as 1
            # broadcast along time axis
            _,_,H,W = vel.shape
            UV = np.zeros((self.tdim_use,2,H,W))
            UV[:,:,:,:] = vel
            
            cc = h5file['C'][()].astype(np.float32)
            cc = cc[None,:,:,:] # add "time" dimension as 1
            # broadcast along time axis
            _,_,H,W = cc.shape
            C = np.zeros((self.tdim_use,1,H,W))
            C[:,:,:,:] = cc

            UVC = np.concatenate([UV,C],axis=1)
            
            rain_Y = rain_all[self.tdim_use:(self.tdim_use*2),:,:,:] # use time tdim_use as X 
            sample = {'past': rain_X, 'future': UVC, 'future_val':rain_Y,
                      'fnames':fnames}
        elif self.mode == "run":
            rain_Y = rain_all[self.tdim_use:(self.tdim_use*2),:,:,:] # use time tdim_use as X 
            sample = {'past': rain_X, 'future': rain_Y,
                      'fnames':fnames}
        h5file.close()
        
        # save

        if self.transform:
            sample = self.transform(sample)

        return sample
