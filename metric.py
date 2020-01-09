import tensorflow.keras.backend as K
import numpy as np

def psnr(x, y, shave=4, luminance=False):
    diff = 0.5 * (x - y)
    '''
    if luminance:
        coeff = np.array([65.738, 129.057, 25.064]).reshape(1, 1, 1, 3) / 256.0
        coeff = K.variable(coeff)
        diff *= coeff
        diff = K.sum(diff, axis=-1)
    '''
    diff = diff[..., shave:-shave, shave:-shave, :]
    #mse = np.mean(diff**2)
    mse = K.mean(diff**2)
    ret = -10.0 * (K.log(mse) / K.log(10.0))
    return ret

def psnr_y(x, y):
    return psnr(x, y, luminance=True)

