import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

## Residual block for the generator 
class ResBlock(nn.Module):
	def __init__(self,in_features):
		super(ResBlock,self).__init__()
		self.conv1=[ nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            		 nn.BatchNorm2d(in_features, 0.8),
            		 nn.PReLU()]
        self.conv2=[ nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            		 nn.BatchNorm2d(in_features, 0.8)]

        self.conv=nn.Sequuential(*conv1,*conv2)

    def forward(self,x):
    	return x + self.conv(x)


### Generator 

class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()

		#first layer of convolution without the residual block
		self.conv1=nn.Sequuential(nn.Conv2d(3,64,9,1,4),
								  nn.Prelu())

		#a serie of residual blocks (8)
		resBlocks=[]
		for i in range(8):
			resBlocks.append(ResBlock(64))
		self.resBlocks=nn.Sequuential(*resBlocks)

		#second convolution after the residual blocks
		self.conv2=nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.BatchNorm2D(64,0.8))

		# upsampling layer with PixelShuffle like stated in the paper
		self.upsampling=nn.Sequential(nn.Conv2d(64,256,3,1,1),
									  nn.BatchNorm2D(256),
									  nn.PixelShuffle(2),
									  nn.PRelu(),
									  nn.Conv2d(64,256,3,1,1),
									  nn.BatchNorm2D(256),
									  nn.PixelShuffle(2),
									  nn.PRelu())
		self.conv3=nn.Sequential(nn.Conv2D(64,3,9,1,4),nn.Tahn())

	def Forward(self,x):
		o1=self.conv1(x)
		o2=self.resBlocks(o1)
		o3=self.conv2(o2)
		o=torch.add(o1,o2)
		o=self.upsampling(o)
		o=self.conv3(o)
		return out


###### Discriminator
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()

	#discriminator reusable convolution block
	def discriminatorBlock(in_filters,out_filters):
		layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
				  nn.BatchNorm2d(out_filters),
				  nn.LeakyReLU(0.2,inplace=True)]
        return layers

    self.model=nn.Seuential(*discriminatorBlock(3,64),
    						*discriminatorBlock(64,128),
    						*discriminatorBlock(128,256),
    						*discriminatorBlock(256,512)
    						nn.ZeroPad2d((1,0,1,0)),
    						nn.Conv2d(512,1,4,1,False)
    						)

    def forward(self,x):
    	return self.model(x)


###### pretrained VGG network for the perceptual loss
class Vgg_Extractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor,self).__init__()
		vgg16_model = vgg16(pretrained=True)
		self.features=nn.Sequential(*list(vgg16_model.features.children())[:15])

	def forward(self,img):
		return self.feature_extractor(img)

