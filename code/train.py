import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
import os

import network_model as _net
from training_params import *

#
# Description:
#  Training code of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

# Train
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Simulated data load path: {datapath_name}")
    print(f"Network save path: {savepath_name}")
    os.makedirs("%s" % savepath_name, exist_ok=True)
    
    ## network
    print(f'==== {"Network loading...":20s} ===')
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
    print(f'==== {"Done":20s} ===')
    
    ### Training with saved data
    print(f'==== {"Training Start...":20s} ===')
    loss_plot = []
    for epoch in range(1,epochs):
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
                    phase = np.load('%s/%s'%(dirName.replace("/input/","/output/"),filename))

                    phase = np.concatenate([phase, phase[:,:1]],axis=1)
                    diff_value = (phase[:,1:]-phase[:,:-1])/2.0
                    for k in range(0,img.shape[0],batch_size):
                        inputs = torch.from_numpy(img[k:k+batch_size ,:,:,:])
                        labels = torch.from_numpy(diff_value[k:k+batch_size ,:])
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

                            loss_set[j] = (loss_set[j]  * input_count  + loss.item()) / (input_count+1)

                            if j == 0:
                                s_outputs = output.cpu().detach()
                                tot_loss = loss.item()
                            else:
                                s_outputs = torch.cat((s_outputs,output.cpu().detach()),dim=1)
                                tot_loss += loss.item()

                    input_count = input_count + img.shape[0]
                    if (iteration_count)%iteration_print == 0:
                        for j in range(len(net_set)):
                            print('Group %d) iter %d) loss: %.10f'%(j, iteration_count, loss_set[j]))
                        loss_plot.append(loss_set)
                        print("Time taken:", datetime.now() - startTime)
                        print('\n-----------------------')
                        startTime = datetime.now()
                        loss_set = np.zeros(len(net_set))
                        input_count = 0

                    if (iteration_count+1)%iteration_lr_step == 0:
                        for j in range(len(net_set)):
                            lr_scheduler_set[j].step(loss_set[j])

        print('Save - epoch %d'%(epoch))
        opt_dict = []
        for i in range(len(net_set)):
            opt_dict.append(optimizer_set[i].state_dict())
        torch.save({'epoch': epoch,
                    'net_dict': net_set.state_dict(),
                    'optimizer_dict': opt_dict,
                    'loss': loss_plot,
                   }, "%s/netdict_first_stage_%d.pth"%(savepath_name, epoch))
    print(f'==== {"Training End...":20s} ===')

if __name__ == '__main__':
    start_time = time.time()
    train()
    print("Total training time : {} sec".format(time.time() - start_time))
