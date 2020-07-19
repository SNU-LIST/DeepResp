import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import rotate as rot
import random

class ImageSet(Dataset):
    def __init__(self,image):
        # image: image data,
        self.len = image.shape[2]
        self.image_data = image

    def __getitem__(self, index):
        img = self.image_data[:,:,index]

        normalized_img = img/(4*np.std(np.abs(img),axis = (0,1),keepdims = True))
        split_img = np.zeros((2,224,224),dtype=np.float32)
        
        split_img[0,:,:] = np.real(normalized_img)
        split_img[1,:,:] = np.imag(normalized_img)

        return torch.from_numpy(split_img)
    
    def __len__(self):
        return self.len
    
class ImagePhaseSet(Dataset):
    def __init__(self,image, phase):
        # image: image data,
        self.len = image.shape[2]
        self.image_data = image
        self.phase_data = phase

    def __getitem__(self, index):
        img = self.image_data[:,:,index]
        phase = self.phase_data[:,index]
        
        normalized_img = img/(4*np.std(np.abs(img),axis = (0,1),keepdims = True))
        split_img = np.zeros((2,224,224),dtype=np.float32)
        
        split_img[0,:,:] = np.real(normalized_img)
        split_img[1,:,:] = np.imag(normalized_img)

        return torch.from_numpy(split_img), torch.from_numpy(phase)
    
    def __len__(self):
        return self.len