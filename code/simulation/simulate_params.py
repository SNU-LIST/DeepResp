image_path = '../data/simulation/train/train_img.npy' # path to artifact-free MR images
                                                      # For train, '../data/simulation/train/train_img.npy'
                                                      # For test, '../data/simulation/test/test_img.npy'
        
resp_path = '../data/simulation/respiration/train_Resp.npy' # path to respiration data
                                                            # For train, '../data/simulation/train/train_resp.npy'
                                                            # For test, '../data/simulation/test/test_resp.npy'
savepath = '../data/train' # path to save simulated data
                           # For train, '../data/train'
                           # For test, '../data/test/'
        

# network paramter
OUTPUT_COUNT = 16 # The number of output points per group
img_size = 224 # Input images size (n x n)
first_channel = 16 # The number of channel of the first convolution layer
initial_lr = 8e-3 # Initial learning rate

# training paramter
epochs = 5 # The number of epochs
batch_size = 100 # bathc size
iteration_lr_step = 50 # The number of interation for one step of learning rate scheduler

iteration_print = 50 # The number of interation to print itermediate results
