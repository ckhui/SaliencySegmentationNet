"""
CODE adapted from: https://github.com/xuebinqin/BASNet/blob/master/model/BASNet.py 
- drop Refine Network
- add Saliency Branch
"""

import torch
import torch.nn as nn
from torchvision import models
# import torch.nn.functional as F
import torch as F
from .resnet_model import *

## Segmentation and Saliency
class SSNet(nn.Module):
    def __init__(self,n_channels, pretrain_resnet = True):
        super(SSNet,self).__init__()

        resnet = models.resnet34(pretrained=pretrain_resnet)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #224
        #stage 2
        self.encoder2 = resnet.layer2 #112
        #stage 3
        self.encoder3 = resnet.layer3 #56
        #stage 4
        self.encoder4 = resnet.layer4 #28

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #14

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = BasicBlock(512,512)
        self.resb6_2 = BasicBlock(512,512)
        self.resb6_3 = BasicBlock(512,512) #7

        ## -------------Bridge -- Segmentation --------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder -- Segmentation --------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(512,256,3,padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(256,128,3,padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)

        # -------------Refine Module-------------
        # self.refunet = RefUnet(1,64)

        ## -------------Bridge -- Saliency--------------

        #stage Bridge
        self.convbg_1_sal = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1_sal = nn.BatchNorm2d(512)
        self.relubg_1_sal = nn.ReLU(inplace=True)
        self.convbg_m_sal = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m_sal = nn.BatchNorm2d(512)
        self.relubg_m_sal = nn.ReLU(inplace=True)
        self.convbg_2_sal = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2_sal = nn.BatchNorm2d(512)
        self.relubg_2_sal = nn.ReLU(inplace=True)

        ## -------------Decoder -- Saliency --------------
        #stage 6d
        self.conv6d_1_sal = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1_sal = nn.BatchNorm2d(512)
        self.relu6d_1_sal = nn.ReLU(inplace=True)

        self.conv6d_m_sal = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m_sal = nn.BatchNorm2d(512)
        self.relu6d_m_sal = nn.ReLU(inplace=True)

        self.conv6d_2_sal = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2_sal = nn.BatchNorm2d(512)
        self.relu6d_2_sal = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1_sal = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1_sal = nn.BatchNorm2d(512)
        self.relu5d_1_sal = nn.ReLU(inplace=True)

        self.conv5d_m_sal = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m_sal = nn.BatchNorm2d(512)
        self.relu5d_m_sal = nn.ReLU(inplace=True)

        self.conv5d_2_sal = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2_sal = nn.BatchNorm2d(512)
        self.relu5d_2_sal = nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1_sal = nn.Conv2d(1024,512,3,padding=1) # 32
        self.bn4d_1_sal = nn.BatchNorm2d(512)
        self.relu4d_1_sal = nn.ReLU(inplace=True)

        self.conv4d_m_sal = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m_sal = nn.BatchNorm2d(512)
        self.relu4d_m_sal = nn.ReLU(inplace=True)

        self.conv4d_2_sal = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2_sal = nn.BatchNorm2d(256)
        self.relu4d_2_sal = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1_sal = nn.Conv2d(512,256,3,padding=1) # 64
        self.bn3d_1_sal = nn.BatchNorm2d(256)
        self.relu3d_1_sal = nn.ReLU(inplace=True)

        self.conv3d_m_sal = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m_sal = nn.BatchNorm2d(256)
        self.relu3d_m_sal = nn.ReLU(inplace=True)

        self.conv3d_2_sal = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2_sal = nn.BatchNorm2d(128)
        self.relu3d_2_sal = nn.ReLU(inplace=True)

        #stage 2d

        self.conv2d_1_sal = nn.Conv2d(256,128,3,padding=1) # 128
        self.bn2d_1_sal = nn.BatchNorm2d(128)
        self.relu2d_1_sal = nn.ReLU(inplace=True)

        self.conv2d_m_sal = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m_sal = nn.BatchNorm2d(128)
        self.relu2d_m_sal = nn.ReLU(inplace=True)

        self.conv2d_2_sal = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2_sal = nn.BatchNorm2d(64)
        self.relu2d_2_sal = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1_sal = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1_sal = nn.BatchNorm2d(64)
        self.relu1d_1_sal = nn.ReLU(inplace=True)

        self.conv1d_m_sal = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m_sal = nn.BatchNorm2d(64)
        self.relu1d_m_sal = nn.ReLU(inplace=True)

        self.conv1d_2_sal = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2_sal = nn.BatchNorm2d(64)
        self.relu1d_2_sal = nn.ReLU(inplace=True)

        ## -------------Side Output--------------
        self.outconvb_sal = nn.Conv2d(512,1,3,padding=1)
        self.outconv6_sal = nn.Conv2d(512,1,3,padding=1)
        self.outconv5_sal = nn.Conv2d(512,1,3,padding=1)
        self.outconv4_sal = nn.Conv2d(256,1,3,padding=1)
        self.outconv3_sal = nn.Conv2d(128,1,3,padding=1)
        self.outconv2_sal = nn.Conv2d(64,1,3,padding=1)
        self.outconv1_sal = nn.Conv2d(64,1,3,padding=1)

    def forward(self,x, segmentation=True, saliency=True):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx) # 256
        h2 = self.encoder2(h1) # 128
        h3 = self.encoder3(h2) # 64
        h4 = self.encoder4(h3) # 32

        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        if segmentation:
            ## -------------Bridge-------------
            hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
            hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
            hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

            ## -------------Decoder -- Segmentation -------------

            hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
            hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
            hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

            hx = self.upscore2(hd6) # 8 -> 16

            hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
            hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
            hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

            hx = self.upscore2(hd5) # 16 -> 32

            hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
            hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
            hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

            hx = self.upscore2(hd4) # 32 -> 64

            hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
            hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
            hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

            hx = self.upscore2(hd3) # 64 -> 128

            hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
            hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
            hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

            hx = self.upscore2(hd2) # 128 -> 256

            hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
            hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
            hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

            ## -------------Side Output-------------
            db = self.outconvb(hbg)
            db = self.upscore6(db) # 8->256

            d6 = self.outconv6(hd6)
            d6 = self.upscore6(d6) # 8->256

            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5) # 16->256

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4) # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3) # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2) # 128->256

            d1 = self.outconv1(hd1) # 256

        ## -------------Refine Module-------------
        # dout = self.refunet(d1) # 256

        if saliency:
            ## -------------Bridge-------------
            hx_sal = self.relubg_1_sal(self.bnbg_1_sal(self.convbg_1_sal(h6))) # 8
            hx_sal = self.relubg_m_sal(self.bnbg_m_sal(self.convbg_m_sal(hx_sal)))
            hbg_sal = self.relubg_2_sal(self.bnbg_2_sal(self.convbg_2_sal(hx_sal)))

            ## -------------Decoder -- Saliency -------------

            hx_sal = self.relu6d_1_sal(self.bn6d_1_sal(self.conv6d_1_sal(torch.cat((hbg_sal,h6),1))))
            hx_sal = self.relu6d_m_sal(self.bn6d_m_sal(self.conv6d_m_sal(hx_sal)))
            hd6_sal = self.relu6d_2_sal(self.bn6d_2_sal(self.conv6d_2_sal(hx_sal)))

            hx_sal = self.upscore2(hd6_sal) # 8 -> 16

            hx_sal = self.relu5d_1_sal(self.bn5d_1_sal(self.conv5d_1_sal(torch.cat((hx_sal,h5),1))))
            hx_sal = self.relu5d_m_sal(self.bn5d_m_sal(self.conv5d_m_sal(hx_sal)))
            hd5_sal = self.relu5d_2_sal(self.bn5d_2_sal(self.conv5d_2_sal(hx_sal)))

            hx_sal = self.upscore2(hd5_sal) # 16 -> 32

            hx_sal = self.relu4d_1_sal(self.bn4d_1_sal(self.conv4d_1_sal(torch.cat((hx_sal,h4),1))))
            hx_sal = self.relu4d_m_sal(self.bn4d_m_sal(self.conv4d_m_sal(hx_sal)))
            hd4_sal = self.relu4d_2_sal(self.bn4d_2_sal(self.conv4d_2_sal(hx_sal)))

            hx_sal = self.upscore2(hd4_sal) # 32 -> 64

            hx_sal = self.relu3d_1_sal(self.bn3d_1_sal(self.conv3d_1_sal(torch.cat((hx_sal,h3),1))))
            hx_sal = self.relu3d_m_sal(self.bn3d_m_sal(self.conv3d_m_sal(hx_sal)))
            hd3_sal = self.relu3d_2_sal(self.bn3d_2_sal(self.conv3d_2_sal(hx_sal)))

            hx_sal = self.upscore2(hd3_sal) # 64 -> 128

            hx_sal = self.relu2d_1_sal(self.bn2d_1_sal(self.conv2d_1_sal(torch.cat((hx_sal,h2),1))))
            hx_sal = self.relu2d_m_sal(self.bn2d_m_sal(self.conv2d_m_sal(hx_sal)))
            hd2_sal = self.relu2d_2_sal(self.bn2d_2_sal(self.conv2d_2_sal(hx_sal)))

            hx_sal = self.upscore2(hd2_sal) # 128 -> 256

            hx_sal = self.relu1d_1_sal(self.bn1d_1_sal(self.conv1d_1_sal(torch.cat((hx_sal,h1),1))))
            hx_sal = self.relu1d_m_sal(self.bn1d_m_sal(self.conv1d_m_sal(hx_sal)))
            hd1_sal = self.relu1d_2_sal(self.bn1d_2_sal(self.conv1d_2_sal(hx_sal)))

            ## -------------Side Output-------------
            db_sal = self.outconvb_sal(hbg_sal)
            db_sal = self.upscore6(db_sal) # 8->256

            d6_sal = self.outconv6_sal(hd6_sal)
            d6_sal = self.upscore6(d6_sal) # 8->256

            d5_sal = self.outconv5_sal(hd5_sal)
            d5_sal = self.upscore5(d5_sal) # 16->256

            d4_sal = self.outconv4_sal(hd4_sal)
            d4_sal = self.upscore4(d4_sal) # 32->256

            d3_sal = self.outconv3_sal(hd3_sal)
            d3_sal = self.upscore3(d3_sal) # 64->256

            d2_sal = self.outconv2_sal(hd2_sal)
            d2_sal = self.upscore2(d2_sal) # 128->256

            d1_sal = self.outconv1_sal(hd1_sal) # 256

            
        seg_map = None
        sal_map = None
        if segmentation:
            seg_map = (F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(db))
        if saliency:
            sal_map = (F.sigmoid(d1_sal), F.sigmoid(d2_sal), F.sigmoid(d3_sal), F.sigmoid(d4_sal), F.sigmoid(d5_sal), F.sigmoid(d6_sal), F.sigmoid(db_sal))
        return seg_map, sal_map
