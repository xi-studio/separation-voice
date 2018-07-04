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

if __name__ == '__main__':
    a,b = default_loader('/data1/littletree/sepvoice_w/0.h5')
    print(a.shape,b.shape)
    
