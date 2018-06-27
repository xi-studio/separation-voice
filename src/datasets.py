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
    imgB = f['out'][:] * 1.0
    f.close()
    return imgA.astype(np.float32), imgB.astype(np.float32)
 


class Radars(data.Dataset):
    def __init__(self):
        super(Radars, self).__init__()
        self.image_list = glob.glob('/data1/littletree/sepvoice/*.h5')

    def __getitem__(self, index):
        path = self.image_list[index]
        imgA,imgB = default_loader(path) # 512x256


        return imgA, imgB

    def __len__(self):

        #return len(self.image_list)
        return 100000


