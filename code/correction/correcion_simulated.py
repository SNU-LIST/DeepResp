import os, time
import numpy as np
import matplotlib.pyplot as plt
from correction_module import correction_img
from correction_params import *

#
# Description:
#  Correction of simulated data with phase errors
#
#  Copyright @ Hongjun An
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : plynt@naver.com
#

def correction():
    print(f'==== {"Correcion start":40s} ===')
    for dirName, subdirList, fileList in sorted(os.walk(img_path)):
            for filename in fileList:
                if ".npy" in filename.lower():
                    img = np.load('%s/%s'%(dirName,filename))
                    ref = np.load('%s/%s'%(ref_phase_path,filename))
                    output = np.load('%s/%s'%(network_phase_path,filename))
                referece_img = correction_img(img,ref,phase_normalized_value)
                corrected_img = correction_img(img,output,phase_normalized_value)
                np.save('%s/%s_%s'%(result_path, 'referece', img), referece_img )
                np.save('%s/%s_%s'%(result_path, 'DeepResp', img), corrected_img)
            break
            print(f'==== {"Done":40s} ===')
            return count
            
if __name__ == '__main__':
    start_time = time.time()
    correction()
    print("Total Correction time : {} sec".format(time.time() - start_time))
