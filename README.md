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

![](results/ali.png)
