"""
Returns the reference image, undersamples k-space data, coil sensitivity maps, and mask 

Jyothi Rikhab Chand, 2024
"""


import pickle
import numpy as np
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def preload(path, start_sub = 0, num_sub_per_type = 2, acc = 4.0, num_sl = 10, contrast_type =1):
    """
    Loads the k-space data and CSM from the given path

    path: data_path
    start_sub & num_sub_per_type: total number of subjects to load
    acc: acceleration factor
    num_sl: number of slices per subject
    contrast_type: To select the contrast type from FastMRI brain data. 
                   For the example data default is 1 as there is only one contrast

    return: k-space data, CSM, Mask, and index
    
    """

    
    subdirs = sorted(os.listdir(path))
    train_ksp, train_csm, labels = None, None, None
    subdir = subdirs[contrast_type]
    fnames = [filename for filename in sorted(os.listdir(path+subdir)) if filename.endswith('.pickle')]
    print(subdir, '- loading', num_sub_per_type, 'of', len(fnames), 'subjects')
        
    subpath = os.path.join(path, subdir)
    train_fnames = fnames[start_sub:start_sub+num_sub_per_type]
        
    for j, train_fname in enumerate(train_fnames):
        with open(os.path.join(subpath, train_fname), 'rb') as f:
            ksp, csm = pickle.load(f)
            ksp, csm = ksp[:num_sl], csm[:num_sl]
            if j==0:
                train_ksp = torch.tensor(ksp)
                train_csm = torch.tensor(csm)
                labels = torch.ones(ksp.shape[0],)*contrast_type
            else:
                train_ksp = torch.cat((train_ksp, torch.tensor(ksp)))
                train_csm = torch.cat((train_csm, torch.tensor(csm)))
                labels = torch.cat((labels, torch.ones(ksp.shape[0],)*contrast_type))
            print('ksp:', ksp.shape, '\tcsm:', csm.shape)
        
    # print('ksp:', train_ksp.shape, '\ncsm:', train_csm.shape, '\nlabels:', labels.shape,)
    
    if acc == 0:
        mask = torch.ones_like(train_ksp)
    elif acc != None:
        mask_filename = f'poisson_mask_2d_acc{acc:.1f}_320by320.npy'
        mask = np.load(mask_filename).astype(np.complex64)  
        mask = torch.tensor(np.tile(mask, [train_ksp.shape[0],train_ksp.shape[1],1,1]))
    else:
        mask = None
    
    labels_key = dict(enumerate([subdir.split('_')[0] for subdir in subdirs]))
    print(f"Loaded dataset of {train_ksp.shape[0]} slices\n")
    
    return train_ksp, train_csm, mask, labels.long(), labels_key





def preprocess(ksp, csm, mask):   
    """
    return: reference image, undersampled k-space data, CSM, mask
    """
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
    us_ksp = ksp * mask
    
    return org, us_ksp.type(torch.complex64), csm.type(torch.complex64), mask




class DataGenBrain(Dataset):
    """
    Dataset class for the MRI dataset
    
    start_sub & num_sub: total number of subjects to load
    acc: acceleration factor
    """
    def __init__(self,  start_sub=0, num_sub=2, device=None, acc=4.0, data_path='Example_data/'):
        
        self.path = data_path
        self.start_sub = start_sub
        self.num_sub = num_sub
        self.device = device
        self.acc = acc
        self.ksp, self.csm, self.msk, self.labels, self.labels_key = preload(self.path, self.start_sub, self.num_sub, acc=acc)
        self.org, self.us_ksp, self.csm, self.msk = preprocess(self.ksp, self.csm, self.msk)


        
    def __len__(self):
        return self.org.size()[0]
    
        
    def __getitem__(self, i):
        return self.org[i:i+1].to(self.device), self.us_ksp[i:i+1].to(self.device), self.csm[i:i+1].to(self.device), self.msk[i:i+1].to(self.device), self.labels[i:i+1].to(self.device)
    
    def get_noisy(self, i, noise_eps=0.):
        us_ksp = self.us_ksp[i:i+1] 
        msk = self.msk[i:i+1]
        scale = 1/torch.sqrt(torch.tensor(2.))
        us_ksp = us_ksp + msk*(torch.randn(us_ksp.shape)+1j*torch.randn(us_ksp.shape))*scale*noise_eps
        
        return self.org[i:i+1].to(self.device), us_ksp.to(self.device), self.csm[i:i+1].to(self.device), msk.to(self.device), self.labels[i:i+1].to(self.device)
    
    