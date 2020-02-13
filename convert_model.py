import argparse
import data
import model

import numpy as np
import imageio

import tensorflow as tf
from tensorflow import lite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_from', type=str, default='models/deblur.hdf5')
    parser.add_argument('-s', '--save_to', type=str, default='models/deblur.tflite')
    parser.add_argument('-d', '--depth', type=int, default=4)
    parser.add_argument('-t', '--test', type=str, default='example/input.png')
    parser.add_argument('-o', '--optimize', type=str, default='')
    parser.add_argument('-q', '--quantize', type=str, default='')
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

    representative = 'REDS/{}/train_blur'
    if h == 256 and w == 256:
        representative = representative.format('train_crop')
    else:
        representative = representative.format('train')

    if cfg.depth == 4:
        net = model.Baseline(h, w)
    else:
        net_class = getattr(model, 'Small{}'.format(cfg.depth))
        net = net_class(h, w)

    net.build(input_shape=(None, h, w, c))
    net.load_weights(cfg.load_from)
    # Make a dummy prediction to get the input shape
    net.predict(test_input, batch_size=1)
    net.summary()

    # Convert to the TFLite model
    converter = lite.TFLiteConverter.from_keras_model(net)
    if cfg.optimize == 'weight':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    '''
    elif 'integer' in cfg.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Dataset for tuning
        converter.representative_dataset = None
        if 'full' in cfg.quantize:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
    elif 'fp16' in cfg.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    '''
    lite_model = converter.convert()
    with open(cfg.save_to, 'wb') as f:
        f.write(lite_model)

if __name__ == '__main__':
    main()
