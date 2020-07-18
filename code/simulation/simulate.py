import torch
import numpy as np
from datetime import datetime
import os
import time
import data_simulator
from simulate_params import *

#
# Description:
#  Training code of DeepResp
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

# simulate
def simulate():
    print(f'==== {"Simulate Start...":20s} ===')
    dataset = data_simulator.Simulator( np.load(image_path), np.load(resp_path),resp_params, TR, SNR, amp, const, isrot, isflip)
    loader = torch.utils.data.DataLoader(dataset, batch_size = data_size, shuffle=True, drop_last=True, num_workers=0)

    os.makedirs("%s/input" % (savepath), exist_ok=True)
    os.makedirs("%s/output" % (savepath), exist_ok=True)

    startTime = datetime.now()
    count = 0
    while count < data_count:
        for i, data in enumerate(loader, 0):
            inputs,phase = data
            np.save('%s/input/%d.npy'%(savepath,count),inputs)
            np.save('%s/output/%d.npy'%(savepath,count), phase)
            count = count + 1
        print("DataCount: %d, Time taken:"%(count,datetime.now() - startTime))
        
if __name__ == '__main__':
    start_time = time.time()
    simulate()
    print("Total Simulation time : {} sec".format(time.time() - start_time))
        
