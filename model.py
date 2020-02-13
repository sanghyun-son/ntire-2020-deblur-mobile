import os
from os import path

from tensorflow import keras
from tensorflow.keras import layers


class Baseline(keras.Model):
    '''
    Baseline residual network with global and local skip connections.

    Args:
        img_h (int): Height of the input image.
        img_w (int): Width of the input image.
        n_colors (int, default=3): The number of the color channels.
        n_feats (int, default=64): The number of the intermediate features.
    '''

    def __init__(self, img_h, img_w, n_colors=3, n_feats=64):
        super().__init__()
        shape = (img_h, img_w, n_colors)
        self.conv_in = layers.Conv2D(
            n_feats, 3, input_shape=shape, padding='same'
        )
        self.res_1 = Residual(n_feats)
        self.res_2 = Residual(n_feats)
        self.res_3 = Residual(n_feats)
        self.res_4 = Residual(n_feats)
        self.conv_out = layers.Conv2D(n_colors, 3, padding='same')

    def call(self, x):
        res = self.conv_in(x)
        res = self.res_1(res)
        res = self.res_2(res)
        res = self.res_3(res)
        res = self.res_4(res)
        res = self.conv_out(res)
        return x + res


class Small0(keras.Model):
    '''
    Baseline residual network with global and local skip connections.

    Args:
        img_h (int): Height of the input image.
        img_w (int): Width of the input image.
        n_colors (int, default=3): The number of the color channels.
        n_feats (int, default=64): The number of the intermediate features.
    '''

    def __init__(self, img_h, img_w, n_colors=3, n_feats=64):
        super().__init__()
        shape = (img_h, img_w, n_colors)
        self.conv_in = layers.Conv2D(
            n_feats, 3, input_shape=shape, padding='same'
        )
        self.relu = ReLU()
        self.conv_out = layers.Conv2D(n_colors, 3, padding='same')

    def call(self, x):
        res = self.conv_in(x)
        res = self.relu(res)
        res = self.conv_out(res)
        return x + res


class Small1(keras.Model):
    '''
    Baseline residual network with global and local skip connections.

    Args:
        img_h (int): Height of the input image.
        img_w (int): Width of the input image.
        n_colors (int, default=3): The number of the color channels.
        n_feats (int, default=64): The number of the intermediate features.
    '''

    def __init__(self, img_h, img_w, n_colors=3, n_feats=64):
        super().__init__()
        shape = (img_h, img_w, n_colors)
        self.conv_in = layers.Conv2D(
            n_feats, 3, input_shape=shape, padding='same'
        )
        self.res_1 = Residual(n_feats)
        self.conv_out = layers.Conv2D(n_colors, 3, padding='same')

    def call(self, x):
        res = self.conv_in(x)
        res = self.res_1(res)
        res = self.conv_out(res)
        return x + res


class Small2(keras.Model):
    '''
    Baseline residual network with global and local skip connections.

    Args:
        img_h (int): Height of the input image.
        img_w (int): Width of the input image.
        n_colors (int, default=3): The number of the color channels.
        n_feats (int, default=64): The number of the intermediate features.
    '''

    def __init__(self, img_h, img_w, n_colors=3, n_feats=64):
        super().__init__()
        shape = (img_h, img_w, n_colors)
        self.conv_in = layers.Conv2D(
            n_feats, 3, input_shape=shape, padding='same'
        )
        self.res_1 = Residual(n_feats)
        self.res_2 = Residual(n_feats)
        self.conv_out = layers.Conv2D(n_colors, 3, padding='same')

    def call(self, x):
        res = self.conv_in(x)
        res = self.res_1(res)
        res = self.res_2(res)
        res = self.conv_out(res)
        return x + res


class Small3(keras.Model):
    '''
    Baseline residual network with global and local skip connections.

    Args:
        img_h (int): Height of the input image.
        img_w (int): Width of the input image.
        n_colors (int, default=3): The number of the color channels.
        n_feats (int, default=64): The number of the intermediate features.
    '''

    def __init__(self, img_h, img_w, n_colors=3, n_feats=64):
        super().__init__()
        shape = (img_h, img_w, n_colors)
        self.conv_in = layers.Conv2D(
            n_feats, 3, input_shape=shape, padding='same'
        )
        self.res_1 = Residual(n_feats)
        self.res_2 = Residual(n_feats)
        self.res_3 = Residual(n_feats)
        self.conv_out = layers.Conv2D(n_colors, 3, padding='same')

    def call(self, x):
        res = self.conv_in(x)
        res = self.res_1(res)
        res = self.res_2(res)
        res = self.res_3(res)
        res = self.conv_out(res)
        return x + res


class Residual(keras.Model):
    '''
    A simple residual block without batch normalization.

    Args:
        n_feats (int): The number of the intermediate features
    '''

    def __init__(self, n_feats):
        super().__init__()
        args = [n_feats, 3]
        self.conv1 = layers.Conv2D(*args, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(*args, padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    patch_size = 128
    m = Baseline(patch_size, patch_size)
    m.build(input_shape=(None, patch_size, patch_size, 3))
    os.makedirs('models', exist_ok=True)
    m.save_weights(path.join('models', 'dummy_deblur.hdf5'))
