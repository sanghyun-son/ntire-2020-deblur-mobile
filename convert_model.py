import argparse
import data
import model

import numpy as np
import imageio

import tensorflow as tf
from tensorflow import lite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', type=str, default='models/deblur.hdf5')
    parser.add_argument('--save_to', type=str, default='models/deblur.tflite')
    parser.add_argument('--test', type=str, default='example/input.png')
    cfg = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Prepare the test input
    test_input = imageio.imread(cfg.test)
    test_input = data.normalize(test_input)
    test_input = np.expand_dims(test_input, axis=0)
    _, h, w, c = test_input.shape

    net = model.Baseline(h, w)
    net.build(input_shape=(None, h, w, c))
    net.load_weights(cfg.load_from)
    # Make a dummy prediction to get the input shape
    net.predict(test_input, batch_size=1)
    net.summary()

    # Convert to the TFLite model
    converter = lite.TFLiteConverter.from_keras_model(net)
    lite_model = converter.convert()
    with open(cfg.save_to, 'wb') as f:
        f.write(lite_model)

if __name__ == '__main__':
    main()
