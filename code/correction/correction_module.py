import numpy as np
def correction_img(img, phase, const):
    k_img = np.fft.fftshift(np.fft.fft2(img,axes=(0,1)),axes=(0,1))
    phase = np.expand_dims(phase, axis=0) * 2j * np.pi * const

    res_k_img = k_img * np.exp(1j *phase)

    corrected_img = np.fft.ifft2(np.fft.ifftshift(res_k_img,axes=(0,1)),axes=(0,1))

    return corrected_img
