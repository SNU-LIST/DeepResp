networkpath_name = '../saved_models/netdict_stage2_4_1.pth' # path of the trained network
datapath_name = '../data/evaluation/simulated/input' # path of evaluation dataset
                                               # For simulated data, '../data/evaluation/simulated/input'
                                               # For in-vivo data, '../data/evaluation/invivo/input'
resultpath_name = '../data/evaluation/simulated/prediction'# path for result

# network paramter
phase_normalized_value = 2 * 0.05 # (simulation paramter) const: scale of phase error to normalize (frequency shifht (Hz) * TE (sec))
batch_size = 200 # batch size