import tensorflow.keras.backend as K
import numpy as np

def psnr(x, y, shave=0, luminance=False, keep_range=False):
    '''
    Args:

    Return:
    '''
    if keep_range:
        diff = (x - y) / 255
    else:
        diff = (x - y) / 2
    '''
    if luminance:
        coeff = np.array([65.738, 129.057, 25.064]).reshape(1, 1, 1, 3) / 256.0
        coeff = K.variable(coeff)
        diff *= coeff
        diff = K.sum(diff, axis=-1)
    '''
    if shave > 0:
        diff = diff[..., shave:-shave, shave:-shave, :]

    mse = K.mean(diff**2)
    ret = -10.0 * (K.log(mse) / K.log(10.0))
    return ret

def psnr_y(x, y):
    return psnr(x, y, luminance=True)

def psnr_full(x, y):
    return psnr(x, y, keep_range=True)

def psnr_np(x, y, shave=0, luminance=False):
    diff = x.astype(np.float32) - y.astype(np.float32)
    if shave > 0:
        diff = diff[..., shave:-shave, shave:-shave, :]

    diff = diff / 255
    mse = np.mean(diff**2)
    ret = -10.0 * (np.log(mse) / K.log(10.0))
    return ret

