savepath_name = '../saved_models' # path to save the trained network
datapath_name = '../data/train' # path of training data set

################first stage########################
# network paramter in the first stage
OUTPUT_COUNT = 16 # The number of output points per group
img_size = 224 # Input images size (n x n)
first_channel = 16 # The number of channel of the first convolution layer
initial_lr = 8e-3 # Initial learning rate

# training paramter in the first stage
epochs = 4 # The number of epochs
batch_size = 100 # batch size
iteration_lr_step = 2 # The number of interation for one step of learning rate scheduler (patience: 10)

iteration_print = 1 # The number of interation to print itermediate results

################second stage#######################
# network paramter in the second stage
first_channel2 = 64 # The number of channel of the first convolution layer
initial_lr2 = 8e-3 # Initial learning rate

# training paramter in the second stage
epochs2 = 1 # The number of epochs
batch_size2 = 100 # batch size
iteration_lr_step2 = 10 # The number of interation for one step of learning rate scheduler (patience: 10)

iteration_print2 = 10 # The number of interation to print itermediate results