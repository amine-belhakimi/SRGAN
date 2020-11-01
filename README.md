# Implementation of SRGAN (Super-Resolution GAN) 
this repo contains my implementation of the (SRGAN) presented in the paper ([Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802.pdf)) 

## Authors
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

## Architecture 

![](figures/srgan.png)

in this implementation i replaced batch-normalisation layer in the Instance-Normalisation layer 
and used a smaller patch GAN discriminator 

in the first attempt i used Relu instead of PRelu activation function then did the same thing as the paper by using PRelu

### Training
since i don't have good hardware to train such large model (i trained for a couple of epochs and it took a long time) 
i used instead a pretrained model [link]() to test the SRGAN 

## Results

i tested on some images of Mohamed Ali profiles, results are shown bellow, the SRGAN architecture is able to create very detailled images 4 time the size of the initial image

![](data/ali.jpg)
![](results/res_0000.png)

the second image

![](data/ali.png)
![](results/res_0001.png)

## Generalisation

we can see that the results are very good but there is some artefactes and deformation in the eyes,this is primarily because of the training data, 
to generalize to a larger types of images we must train our network with different dataset in different contexte, this will allow a better generalisation to different images in different contexts 

![](figures/video_fusion.png)
## Videos 

we can use the same architecture to unhance the quality of a video by using a 3D convolution, but a major problem with videos is the temporel dependecie between frames.
one solution is proposed by Karpathy et al.
using stacked frames on top of eachothers instead of enhancing one frame at the time, another solution is to use LSTMs (Long Short Term Momory) recurent neural networks.

# References
the code for this projet is based on : [link](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix) 

#### papers 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)
[Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)
[ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
