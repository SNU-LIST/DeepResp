import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os, time

import network_model_stage1 as _net
import network_model_stage2 as _net2
from evaluation_params import *
import data_loader as dl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
# Description:
#  Evaluation for simulated data code of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

def eval_simualted():
    ## network
    print(f'==== {"Network loading...":40s} ===')

    checkpoint = torch.load(networkpath_name)

    img_size = checkpoint['1']['net_params']['img_size']
    first_channel1 = checkpoint['1']['net_params']['first_channel']
    first_channel2 = checkpoint['2']['net_params']['first_channel']
    OUTPUT_COUNT = checkpoint['1']['net_params']['OUTPUT_COUNT']
    net_set = nn.ModuleList()
    for i, num in enumerate(range(0,img_size,OUTPUT_COUNT)):
        net_set.append(_net.Net(num, num+OUTPUT_COUNT-1, img_size, first_channel1, OUTPUT_COUNT))

    for i in range(len(net_set)):
        net_set[i].load_state_dict(checkpoint['1']['net_dict'][i])
        net_set[i].to(device)
        if torch.cuda.device_count() > 1:
            net_set[i] = nn.DataParallel(net_set[i])

    AccumNet = _net2.CAE(first_channel2)
    AccumNet.load_state_dict(checkpoint['2']['net_dict'])
    AccumNet.to(device)
    if torch.cuda.device_count() > 1:
        AccumNet = nn.DataParallel(AccumNet)

    print(f'==== {"Done":40s} ===')
    print(f'==== {"Evaluation start...":40s} ===')

    net_set.eval()
    AccumNet.eval()
    count = 0
    for dirName, subdirList, fileList in sorted(os.walk(datapath_name)):
        for filename in fileList:
            if ".npy" in filename.lower():
                img= dl.ImageSet(np.load('%s/%s'%(dirName,filename)))
                loader = torch.utils.data.DataLoader(img, batch_size = batch_size, shuffle=False, num_workers=0)
                for i, inputs in enumerate(loader, 0):
                    count = count +  inputs.shape[0]
                    inputs = inputs.to(device)
                    if device == torch.device('cuda'):
                        inputs = inputs.type(torch.cuda.FloatTensor)
                    k_data = torch.fft(inputs.permute(0,2,3,1),2)
                    with torch.no_grad():
                        for j in range(len(net_set)):
                            output = net_set[j](inputs,k_data)
                            if j == 0:
                                s_outputs = output.cpu().detach()
                            else:
                                s_outputs = torch.cat((s_outputs,output.cpu().detach()),dim=1)
                    outputs = AccumNet(s_outputs.unsqueeze(1)).squeeze()
                    if i == 0:
                        tot_outputs = outputs.cpu().detach()
                    else:
                        tot_outputs = torch.cat((tot_outputs,outputs.cpu().detach()),dim=0)
                np.save('%s/%s'%(resultpath_name, filename), tot_outputs.cpu().detach().numpy())
    print(f'==== {"Done":40s} ===')
    return count

if __name__ == '__main__':
    start_time = time.time()
    print(f"Simulated data load path: {datapath_name}")
    print(f"Saved network path: {networkpath_name}")
    print(f"Result path: {resultpath_name}")
    os.makedirs("%s" % resultpath_name, exist_ok=True)
    if len(sorted(os.walk(datapath_name))) == 0:
        print("No data!!")
    else:
        tot_count = eval_simualted()
        print(f"Total evaluation time : {time.time() - start_time} sec for {tot_count} images")
