from tensorflow.keras import models
from tensorflow.keras import layers

class SRCNN(models.Sequential):

    def __init__(self, img_h, img_w, n_colors=3):
        super().__init__()
        kwargs = {'padding': 'same', 'activation': 'relu'}
        input_shape = (img_h, img_w, n_colors)
        self.add(layers.Conv2D(64, 9, input_shape=input_shape, **kwargs))
        self.add(layers.Conv2D(32, 1, **kwargs))
        self.add(layers.Conv2D(3, 5, padding='same'))

