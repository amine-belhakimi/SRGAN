import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataloader
from torchvision.utils import save_image
from models import *
from dataset import *

device = "cpu"


generator=Generator()
discriminator =Discriminator()
features=Vgg_Extractor()


criterion_Gan=torch.nn.MSELoss()
criterion_percept= torch.nn.L1Loss()

generator=generator.to(device)
discriminator=discriminator.to(device)
features=features.to(device)

criterion_Gan=criterion_Gan.to(device)
criterion_percept=criterion_percept.to(device)

optimizer_G=torch.optim.Adam(generator.parameters(),lr=0.001,betas=(0.99,0.99))
optimizer_D=torch.optim.Adam(generator.parameters(),lr=0.001,betas=(0.99,0.99))

Tensor=torch.cuda.FloatTensor if device=="gpu" else torch.Tensor

dataloader=Dataloader(ImagesDataset("data/HRimages",(125,125)),
						batch_size=3,
						shuffle=True)


def extractor(real,fake):
	f_HR=features(real)
	f_LR=features(fake)
	return f_HR,f_LR


print("start training")
for epoch in range(opt.epoch,opt.n_epochs):
	print("epoch :",epoch)
	for i,img in enumerat(dataloader):

		img_lr=Variable(img["lr"].type(Tensor))
		imgs_hr = Variable(img["hr"].type(Tensor))

		D_real=Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        D_fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)


       	########################
        optimizer_G.zero_grade()

        gen_hr=generator(img_lr)

        loss_GAN=criterion_Gan(discriminator(gen_hr),D_valid)

        f_HR,f_LR=extractor(img_hr,gen_hr)
        vgg_loss=criterion_percept(f_HR,f_LR)

        loss_G= 1e-3 * loss_GAN + vgg_loss

        loss_G.backward()
        optimizer_G.step()


       #######################""

       optimizer_D.zero_grade()

       loss_real=criterion_Gan(discriminator(imgs_hr),D_valid)
       loss_fake=criterion_Gan(discriminator(img_lr),D_fake)

       loss_D=(loss_real,loss_fake)/2

       loss_D.backward()
       optimizer_D.step()




       if epoch % print_interval==0:
       		print("saving sample image of train to visualize")
       		print("loss G:", Loss_G)
       		save_image(gen_hr,"image_"+epoch+".png")


    if epoch % save_interval ==0:
    	torch.save(generator.state_dict(),"generator_"+epoch+".pth")
    	torch.save(discriminator.state_dict(),"generator_"+epoch+".pth")
