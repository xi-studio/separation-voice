import glob
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random
import h5py

def default_loader(path):
    f = h5py.File(path,'r')
    imgA = f['in'][:]
    f.close()
    return imgA
 

class Voice(data.Dataset):
    def __init__(self,filepath):
        super(Voice, self).__init__()
        self.image_list = glob.glob(filepath + '/*.h5')

    def __getitem__(self, index):
        path = self.image_list[index]
        imgA = default_loader(path)

        return imgA

    def __len__(self):

        return len(self.image_list)


