image_path = '../../data/acquired_data_for_simulation/test/test_img_mini.npy' # path to artifact-free MR images
                                                      # For train, '../../data/acquired_data_for_simulation/train/train_img_mini.npy'
                                                      # For test, '../../data/acquired_data_for_simulation/test/test_img_mini.npy'
        
resp_path = '../../data/acquired_data_for_simulation/test/test_resp_mini.npy' # path to respiration data
                                                            # For train, '../../data/acquired_data_for_simulation/train/train_resp_mini.npy'
                                                            # For test, '../../data/acquired_data_for_simulation/test/test_resp_mini.npy'
savepath = '../../data/evaluation/simulated' # path to save simulated data
                           # For train, '../../data/train'
                           # For test, '../../data/evaluation/simulated'
        

##Parameter
const = 2 * 0.05 # const: scale of phase error to normalize (frequency shifht (Hz) * TE (sec))
TR = 1.2 # TR (sec)(sampling period of respiration data to reformat into phase errors)
isrot = 10 # rotation angle for image augmentation
isflip = 0.5 # probablity to apply horizontal flipping for image augmentation
amp = [0.05, 1] # output scale range [min, max]
SNR = [50,200] # SNR range [min, max]
resp_params = [500, 390] # rparamter of respiration data [sampling rate (Hz), measuretime(sec)]

data_size = 500 # The number of images per one npy file
data_count = 10 # The number of npy files
