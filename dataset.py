import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import glob


class ImagesDataset(Dataset):
  def __init__(self,root,shape):
    height,width=shape

    #the transform for the lower resolution image
    self.transform_lower=transform.Compose([
      transforms.Resize((height//4,width//4),Image.BICUBIC)
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
      ])

    #the transform for the higher resolution image
    self.transform_higher=transform.Compose([
      transfomr.Resize((height,width),Image.BICUBIC)
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
      ])

    #list of the images in the folder
    self.files=sorted(glob.glob(root + "/*.*"))

  def __getitem__(self,index):
    #get one image of index 'index'
    img=Image.open(self.files[index % len(self.files)])
    #transform the images
    lr_image=self.transform_lower(img)
    hr_image=self.transform_higher(img)

    return {"lr":lr_image,"hr":hr}

  def__len__(self):
    return len(self.files)