import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Description:
#  Modified ResNet in the first stage of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

def weights_inititialize(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class CAE(nn.Module):
    def __init__(self, first_c):
        super(CAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,first_c,3,stride = 2,padding=0),
            nn.BatchNorm1d(first_c),
            nn.ReLU(True),
            nn.Conv1d(first_c ,first_c * 2,3,stride = 2,padding=1),
            nn.BatchNorm1d(first_c * 2),
            nn.ReLU(True),
            nn.Conv1d(first_c * 2,first_c * 4,3,stride = 2,padding=1),
            nn.BatchNorm1d(first_c * 4),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(first_c * 4, first_c * 2, 4,stride = 2,padding=1),
            nn.BatchNorm1d(first_c * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(first_c * 2, first_c , 4,stride = 2,padding=1),
            nn.BatchNorm1d(first_c),
            nn.ReLU(True),
            nn.ConvTranspose1d(first_c, first_c//2, 4,stride = 2,padding=1),
            nn.BatchNorm1d(first_c//2),
            nn.ReLU(True),
            nn.Conv1d(first_c//2,1,1),
        )
    def forward(self,x):
        x = F.pad(x,(1,1),'circular')
        x = self.encoder(x)
        x = self.decoder(x)
        return x
