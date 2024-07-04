import torch
import torch.nn as nn
from nets.part import *
from nets.resnet import resnet50
import numpy as np


class DR_UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.dialtedconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1)
        self.dialtedconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2,dilation=2)
        self.dialtedconv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=5,dilation=5)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.dialtedconv1(x)
        x1 = self.Relu(x1)
        x1 = self.dialtedconv2(x1)
        x1 = self.Relu(x1)
        x1 = self.dialtedconv3(x1)
        x1 = self.Relu(x1)
        return x1
    

class GFRF(nn.Module):
    def __init__(self, in_size, out_size,in_size2):
        super(GFRF, self).__init__()
        self.dialtedconv = DR_UP(in_size,out_size)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)
        self.conv = nn.Conv2d(out_size,out_size,kernel_size=1,stride=1)
        self.conv1 = nn.Conv2d(in_size2,out_size,kernel_size=1,stride=1)


    def forward(self, inputs1, inputs2):
        x1 = inputs1
        x2 = self.up(inputs2)
        outputs = torch.cat([x1,x2],1)
        outputs = self.dialtedconv(outputs)
        x1 = self.conv(x1)
        x2 = self.conv1(x2)
        outputs = outputs + x1 + x2
        return outputs


    
class Unet(nn.Module):
    def __init__(self, num_classes = 2, pretrained = False, backbone = 'resnet50'):
        super(Unet, self).__init__()
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.resnet = resnet50(pretrained = pretrained)
        in_filters  = [320, 768, 1536, 3072]
        out_filters = [64, 256, 512, 1024]
        in_size2 = [256,512,1024,2048]
        # upsampling
        # 64,64,512
        self.up_concat4 = GFRF(in_filters[3], out_filters[3],in_size2[3])
        # 128,128,256
        self.up_concat3 = GFRF(in_filters[2], out_filters[2],in_size2[2])
        # 256,256,128
        self.up_concat2 = GFRF(in_filters[1], out_filters[1],in_size2[1])
        # 512,512,64
        self.up_concat1 = GFRF(in_filters[0], out_filters[0],in_size2[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(64, num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

