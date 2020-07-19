import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os, time
import data_loader as dl
import network_model_stage1 as _net
import network_model_stage2 as _net2
from training_params import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
# Description:
#  Training code of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

## train
def train():
    print(f'==== {"Network in the first stage loading...":40s} ===')
    net_set = nn.ModuleList()
    optimizer_set = []
    lr_scheduler_set = []

    for i, num in enumerate(range(0,img_size,OUTPUT_COUNT)):
        net_set.append(_net.Net(num, num+OUTPUT_COUNT-1, img_size, first_channel, OUTPUT_COUNT))

    for i in range(len(net_set)):
        net_set[i].to(device)
        if torch.cuda.device_count() > 1:
            net_set[i] = nn.DataParallel(net_set[i])
        net_set[i].apply(_net.weights_initialize)
        optimizer_set.append(optim.Adam(net_set[i].parameters(), lr=initial_lr))
        lr_scheduler_set.append(optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_set[i],
                                             mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-3))
    criterion = nn.MSELoss()
    print(f'==== {"Done":40s} ===')
    
    ### Training with saved data (load data)##
    print(f'==== {"First stage training Start...":40s} ===')

    loss_plot = []
    for epoch in range(1,epochs+1):
        print(f'==== epoch {epoch:15d} ===')

        startTime = datetime.now()
        input_count = 0
        loss_set = np.zeros(len(net_set))
        net_set.train(True)
        iteration_count = 0
        for dirName, subdirList, fileList in sorted(os.walk(datapath_name + "/input")):
            for filename in fileList:
                if ".npy" in filename.lower():
                    iteration_count = iteration_count + 1
                    img = np.load('%s/%s'%(dirName,filename))
                    phase = np.load('%s/%s'%(dirName.replace("/input","/output"),filename))

                    phase = np.concatenate([phase, phase[:,:1]],axis=1)
                    diff_value = (phase[:,1:]-phase[:,:-1])/2.
                    
                    dataset= dl.ImagePhaseSet(img, diff_value)
                    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)
                    for i, data in enumerate(loader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        if device == torch.device('cuda'):
                            inputs = inputs.type(torch.cuda.FloatTensor)
                            labels = labels.type(torch.cuda.FloatTensor)
                        k_data = torch.fft(inputs.permute(0,2,3,1),2)
                        
                        for j in range(len(net_set)):
                            optimizer_set[j].zero_grad()
                            output = net_set[j](inputs,k_data)
                            loss = criterion(output, labels[:,OUTPUT_COUNT*j:OUTPUT_COUNT*(j+1)])
                            loss.backward()
                            optimizer_set[j].step()

                            if j == 0:
                                s_outputs = output.cpu().detach()
                            else:
                                s_outputs = torch.cat((s_outputs,output.cpu().detach()),dim=1)
                            loss_set[j] = (loss_set[j]  * i + loss.item()) / (i + 1)
     
                    if (iteration_count+1)%iteration_lr_step == 0:
                        for j in range(len(net_set)):
                            lr_scheduler_set[j].step(loss_set[j])
                            
                    if (iteration_count)%iteration_print == 0:
                        for j in range(len(net_set)):
                            print('Group %d) iter %d) loss: %.10f'%(j, iteration_count, loss_set[j]))
                        loss_plot.append(loss_set)
                        print("Time taken:", datetime.now() - startTime)
                        print('\n-----------------------')
                        startTime = datetime.now()
                        loss_set = np.zeros(len(net_set))
                        input_count = 0

                    

        print('Save - epoch %d'%(epoch))
        opt_dict = []
        net_dict = []
        for i in range(len(net_set)):
            opt_dict.append(optimizer_set[i].state_dict())
            if torch.cuda.device_count() > 1:
                net_dict.append(net_set[i].module.state_dict())
            else:
                net_dict.append(net_set[i].state_dict())
        save_dict = {'1':{'epoch': epoch,
                    'net_dict': net_dict,
                    'optimizer_dict': opt_dict,
                    'loss': loss_plot,
                     'net_params':{'img_size':img_size, 
                                   'first_channel':first_channel,
                                   'OUTPUT_COUNT':OUTPUT_COUNT}}}
        torch.save(save_dict, "%s/netdict_stage1_%d.pth"%(savepath_name, epoch))
    print(f'==== {"First stage training End...":40s} ===')
    
    ## network
    print(f'==== {"Network in the second stage loading...":40s} ===')
    AccumNet = _net2.CAE(first_channel2)
    AccumNet.to(device)
    if torch.cuda.device_count() > 1:
        AccumNet = nn.DataParallel(AccumNet)
    AccumNet.apply(_net2.weights_inititialize)

    optimizer = optim.Adam(AccumNet.parameters(), lr=initial_lr2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                             mode='min',
                                             factor=0.5,
                                             patience=10, verbose=True, threshold=1e-3)

    criterion = nn.MSELoss()
    print(f'==== {"Done":40s} ===')
    
    ### Training with saved data (load data)##
    print(f'==== {"Second stage training Start...":40s} ===')

    loss_plot = []
    for epoch in range(1,epochs2+1):
        print(f'==== epoch {epoch:15d} ===')

        startTime = datetime.now()
        input_count = 0
        loss_set = 0.0
        net_set.eval()
        AccumNet.train(True)
        iteration_count = 0
        for dirName, subdirList, fileList in sorted(os.walk(datapath_name + "/input")):
            for filename in fileList:
                if ".npy" in filename.lower():
                    iteration_count = iteration_count + 1
                    img = np.load('%s/%s'%(dirName,filename))
                    phase = np.load('%s/%s'%(dirName.replace("/input","/output"),filename))
                    
                    dataset= dl.ImagePhaseSet(img, phase)
                    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size2, shuffle=True, num_workers=0)
                    for i, data in enumerate(loader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs, labels = inputs.to(device), labels.to(device)
                        if device == torch.device('cuda'):
                            inputs = inputs.type(torch.cuda.FloatTensor)
                            labels = labels.type(torch.cuda.FloatTensor)
                        k_data = torch.fft(inputs.permute(0,2,3,1),2)
                        with torch.no_grad():
                            for j in range(len(net_set)):
                                output = net_set[j](inputs,k_data)
                                if j == 0:
                                    s_outputs = output.cpu().detach()
                                else:
                                    s_outputs = torch.cat((s_outputs,output.cpu().detach()),dim=1)

                        optimizer.zero_grad()
                        outputs = AccumNet(s_outputs.unsqueeze(1)).squeeze()
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        loss_set = (loss_set * i + loss.item()) / (i + 1)
                        
                    if (iteration_count+1)%iteration_lr_step2 == 0:
                        lr_scheduler.step(loss_set)

                    if (iteration_count)%iteration_print2 == 0:
                        print('iter %d) loss: %.10f'%(iteration_count, loss_set))
                        loss_plot.append(loss_set)
                        print("Time taken:", datetime.now() - startTime)
                        print('\n-----------------------')
                        startTime = datetime.now()
                        loss_set = 0.0
                        input_count = 0

                    

        print('Save - epoch %d'%(epoch))

        if torch.cuda.device_count() > 1:
            net_dict = AccumNet.module.state_dict()
        else:
            net_dict = AccumNet.state_dict()

        save_dict['2'] = {'epoch': epoch,
                'net_dict': net_dict,
                'optimizer_dict': optimizer.state_dict(),
                'loss': loss_plot,
                 'net_params':{'first_channel':first_channel2}}
        torch.save(save_dict, "%s/netdict_stage2_%d_%d.pth"%(savepath_name,save_dict['1']['epoch'], epoch))

    print(f'==== {"Second stage training End...":40s} ===')

if __name__ == '__main__':
    start_time = time.time()
    print(f"Simulated data load path: {datapath_name}")
    print(f"Network save path: {savepath_name}")
    os.makedirs("%s" % savepath_name, exist_ok=True)
    if len(sorted(os.walk(datapath_name + "/input"))) == 0:
        print("No data!!")
    else:
        train()
        print("Total training time : {} sec".format(time.time() - start_time))
