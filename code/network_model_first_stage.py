import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#
# Description:
#  Modified ResNet in the first stage of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#


def weights_initialize(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class First_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(First_block,self).__init__()
        self.conv = nn.Conv2d(in_c,out_c,(1,224),(1,2),padding = 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(3,(2, 1),padding = 1)
        
    def forward(self,x):
        x = F.pad(x,(112,111,0,0),'circular')
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x
        
class Bottleneck_block(nn.Module):
    def __init__(self,in_c,mid_c,out_c,stri=1):
        super(Bottleneck_block,self).__init__()
        self.conv1 = nn.Conv2d(in_c,mid_c,1,1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c,mid_c,3,stride=stri, padding = 0)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c,out_c,1,1)
        self.bn3 = nn.BatchNorm2d(out_c)    
        
        self.convS = nn.Conv2d(in_c,out_c,1,stride=stri)
        self.bnS = nn.BatchNorm2d(out_c)
        
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        
        y = F.pad(y,(1,1,1,1),'circular')
        y = self.conv2(y)
        y = self.bn2(y)
        y = F.relu(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        x = self.convS(x)
        x = self.bnS(x)
        
        out = F.relu(y + x)
        return out
    
class Last_block(nn.Module):
    def __init__(self,in_c,last_c):
        super(Last_block,self).__init__()
        self.in_c = in_c
        self.layer1 = nn.AdaptiveAvgPool2d((1))
        self.layer2 = nn.Linear(in_c,last_c)
    def forward(self,x):
        x = self.layer1(x)
        x = x.view(-1,self.in_c)
        x = self.layer2(x)
        return x
    
class Bottleneck(nn.Module):
    def __init__(self,in_c,block_n,first = False):
        super(Bottleneck,self).__init__()
        if first == True:
            out_c = in_c * 4
            self.block1 = Bottleneck_block(in_c,in_c,out_c,1)
            self.blocks = nn.ModuleList([Bottleneck_block(out_c,in_c,out_c,1) for i in range(block_n-1)])
        else:
            out_c = in_c * 2
            self.block1 = Bottleneck_block(in_c,in_c//2,out_c,2)
            self.blocks = nn.ModuleList([Bottleneck_block(out_c,in_c//2,out_c,1) for i in range(block_n-1)])
    def forward(self,x):
        x = self.block1(x)
        for layer in self.blocks:
            x = layer(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, in_c, first_c, last_c):
        super(ResNet50, self).__init__()
        self.block0 = First_block(in_c,first_c)
        
        self.block1 = Bottleneck(first_c   ,3 ,True)
        self.block2 = Bottleneck(first_c*4 ,4)
        self.block3 = Bottleneck(first_c*8 ,6)
        self.block4 = Bottleneck(first_c*16,3)
        
        self.blockL = Last_block(first_c *32 , last_c)
 
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.blockL(x)
        return x
    
class Bandpassfilter(nn.Module):
    def __init__(self, num_s, num_e,img_size):
        super(Bandpassfilter, self).__init__()
        self.fil = np.zeros((1,img_size,img_size,2), dtype=np.float32)
        self.fil[0,:,num_s:num_e+2,:] = 1
        if num_e+2 > img_size:
            self.fil[0,:,:num_e+2-img_size,:] = 1

        self.fil = np.fft.fftshift(self.fil,axes=(1,2))
        self.fil = torch.tensor(self.fil)        
        self.fil = torch.nn.Parameter(self.fil,requires_grad=False)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        x = x * self.fil
        x = torch.ifft(x,2)
        x = x.permute(0,3,1,2)
        x = self.bn(x)
        return x

class Net(nn.Module):
    def __init__(self, num_s, num_e, img_size, first_c, last_c):
        super(Net, self).__init__()
        self.net = ResNet50(4,first_c,last_c)
        self.BPF = Bandpassfilter(num_s,num_e,img_size)
        
    def forward(self, x, k):
        x = self.net(torch.cat([x, self.BPF(k)],dim=1))
        return x
