import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

from model.models import *
from dataset.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# Loss functions
criterion_GAN = torch.nn.MSELoss()
#criterion_pixelwise = torch.nn.L1Loss()
criterion_pixelwise = torch.nn.MSELoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=2, out_channels=2).to(device)
#discriminator = Discriminator().to(device)


if cuda:
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
    #discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
dataloader = DataLoader(Radars(),batch_size=opt.batch_size, shuffle=True, num_workers=1)

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        A, B = batch
        A = A.to(device)
        B = B.to(device)

        # Model inputs

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(A)
        loss_pixel = criterion_pixelwise(fake_B, B)


        loss_pixel.backward()

        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [pixel: %f ] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_pixel.item(), 
                                                        time_left))

#        # If at sample interval save image
#        if batches_done % opt.sample_interval == 0:
#            sample_images(batches_done)
#
#
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
#        torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, epoch))
torch.save(generator.state_dict(), 'saved_models/%s/generator_end.pth' % (opt.dataset_name))
