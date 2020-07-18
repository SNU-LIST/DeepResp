image_path = '../data/simulation/train/train_img.npy' # path to artifact-free MR images
                                                      # For train, '../data/simulation/train/train_img.npy'
                                                      # For test, '../data/simulation/test/test_img.npy'
        
resp_path = '../data/simulation/respiration/train_Resp.npy' # path to respiration data
                                                            # For train, '../data/simulation/train/train_resp.npy'
                                                            # For test, '../data/simulation/test/test_resp.npy'
savepath = '../data/train' # path to save simulated data
                           # For train, '../data/train'
                           # For test, '../data/test/'
        

##Parameter
const = 0.05 * 2 # const: scale of phase error to normalize (frequency shifht (Hz) * TE (sec))
TR = 1.2 # TR (sec)(sampling period of respiration data to reformat into phase errors)
isrot = 10 # rotation angle for image augmentation
isflip = 0.5 # probablity to apply horizontal flipping for image augmentation
amp = [0.05, 1] # output scale range [min, max]
SNR = [50,200] # SNR range [min, max]
resp_params = [500, 390] # rparamter of respiration data [sampling rate (Hz), measuretime(sec)]

data_size = 100 # The number of images per one npy file
data_count = 100 # The number of npy files
