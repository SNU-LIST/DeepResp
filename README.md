# DeepResp
* The code is for correction of respidation-induced B0 fluctuation artifacts using deep learning (DeepResp)
* Last update : 2020.07.19
* The source data for training can be shared to academic institutions. Request should be sent to snu.list.software@gmail.com. For each request, individual approval from our institutional review board is required (i.e. takes time)
* For more information, refer to the published paper (https://doi.org/10.1016/j.neuroimage.2020.117432)

# References
* DeepResp: Deep learning solution for respiration-induced B0 fluctuation artifacts in multi-slice GRE
H. An, H.-G. Shin, S. Ji, W. Jung, S. Oh, D. Shin, J. Park, J. Lee. DeepResp: Deep learning solution for respiration-induced B0 fluctuation artifacts in multi-slice GRE. Neuroimage. 2021 Jan. v 224. https://www.sciencedirect.com/science/article/pii/S1053811920309174

# Overview
## DeepResp
![grapical_abstract](https://user-images.githubusercontent.com/57519974/87869870-3e197580-c9de-11ea-8092-b0d017abbb82.png)


## Requirements
* Python 3.7

* Pytorch 1.5.1

* NVIDIA GPU (CUDA 10.1) (MultiGPUs are avaliable)


## Data acquisition
* MR images for simulation were acquired at 3T MRI (SIEMENS), which were from below refereces. The images were either zero-padded or cropped in k-space to match the matrix size to 224 × 224. Each image was masked out noises in the background using an intensity threshold to remove artifacts in the background.

  * QSMnet </br>
  _J. Yoon, E. Gong, I. Chatnuntawech, B. Bilgic, J. Lee, W. Jung, J. Ko, H. Jung, K. Setsompop, G. Zaharchuk, E.Y. Kim, J. Pauly, J. Lee.
  Quantitative susceptibility mapping using deep neural network: QSMnet.
  Neuroimage. 2018 Oct;179:199-206. https://www.sciencedirect.com/science/article/pii/S1053811918305378_
  * QSMnet+ </br>
  _W. Jung, J. Yoon, S. Ji, J. Choi, J. Kim, Y. Nam, E. Kim, J. Lee. Exploring linearity of deep neural network trained QSM: QSMnet+.
  Neuroimage. 2020 May; 116619. https://www.sciencedirect.com/science/article/pii/S1053811920301063_


* Respiration data for simulation was acquired with a a temperature sensor (Biopac). The data were sampled at 500 Hz and recorded for 7 sessions, each with 390 seconds. A median filter and a bandpass-filter (passband: 0.1 Hz ~ 1 Hz) were applied to reduce noise.

* MR images for in-vivo experments were acquired at 3T MRI (SIEMENS) using a multi-slice GRE sequence with a navigator echo. The scan parameters were as follows: TR = 1200 ms, TE = 6.9 ms, 15.2 ms, 20.5 ms, 25.7 ms, 31.0 ms, 36.3 ms, and 41.5 ms for the images, 55.0 ms for the navigator, flip angle = 70°, bandwidth = 260 Hz/pixel, FOV = 224 × 224 mm2, in-plane resolution = 1 × 1 mm2, slice thickness = 2 mm, distance factor = 20%, and 18 slices for 9 subjects and 16 slices for 1 subject.


## Simulation
* The source code for simulation generates the simulated respiration-corrupted images with the MR images and the respiration data.
* MR images : (Height x Width x slices) complex numpy data,  Respiration data : (Subjects x data sample) float numpy data, 
* Results: the complex-valued numpy images ( read-out x phase-encoding x slice ) are generated.

## Training
* The source code for training. The training performed with the saved data from the simulation.

## Evaluation
* The source code for evaluation of the trained neural networks.
* The evaluation can be performed with the simulated data and the in-vivo data.
* Results: networks-generated phase errors

## Correction
* The source code for correction of corrupted images using phase errors (network-generated or reference)
* Results: DeepResp-corrected images, reference-corrected images

