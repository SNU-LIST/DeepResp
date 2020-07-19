import torch
import numpy as np
from datetime import datetime
import os
import time
import data_simulator
from simulate_params import *


#
# Description:
#  Simulation to generate respiration-corrupted images of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

# simulate
def simulate():
    print(f'==== {"Simulation Start...":20s} ===')
    img = np.load(image_path)
    resp = np.load(resp_path)
    if (img.shape[2]<data_size):
        img = np.repeat(img,data_size//img.shape[2]+1,axis=2)
        
    dataset = data_simulator.Simulator( img,resp,resp_params, TR, SNR, amp, const, isrot, isflip)
    loader = torch.utils.data.DataLoader(dataset, batch_size = data_size, shuffle=True, drop_last=True, num_workers=0)

    os.makedirs("%s/input" % (savepath), exist_ok=True)
    os.makedirs("%s/output" % (savepath), exist_ok=True)

    startTime = datetime.now()
    count = 1
    while count <= data_count:
        for i, data in enumerate(loader, 0):
            inputs, phase = data
            s_inputs = inputs[:,0,:,:].numpy() + 1j*inputs[:,1,:,:].numpy()
            s_inputs = np.transpose(s_inputs,(1,2,0))
            phase = np.transpose(phase,(1,0))
            np.save('%s/input/%d.npy'%(savepath,count),s_inputs)
            np.save('%s/output/%d.npy'%(savepath,count), phase)
            count = count + 1
            if count > data_count: break
        print(f"DataCount: {count:d}, Time taken: {datetime.now() - startTime}")

if __name__ == '__main__':
    start_time = time.time()
    simulate()
    print("Total Simulation time : {} sec".format(time.time() - start_time))