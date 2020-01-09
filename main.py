from os import path
import argparse

import tensorflow as tf
from tensorflow import lite
from tensorflow.keras import callbacks

import model
import data
import metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, defalut=128)
    parser.add_argument('--batch_size', type=int, defalut=16)
    parser.add_argument('--epochs', type=int, defalut=100)
    parser.add_argument('--exp_name', type=str, defalut='baseline')
    parser.add_argument('--save_to', type=str, defalut='deblur.tflite')
    cfg = parser.parse_args()

    # For checking the GPU usage
    #tf.debugging.set_log_device_placement(True)

    dataset_train = data.REDS(
        cfg.batch_size, patch_size=cfg.patch_size, train=True
    )
    # We have 3,000 validation frames.
    # Note that each frame will be center-cropped for the validation.
    dataset_val = data.REDS(20, patch_size=cfg.patch_size, train=False)

    net = model.SRCNN(cfg.patch_size, cfg.patch_size)
    net.compile(optimizer='adam', loss='mse', metrics=[metric.psnr])
    net.summary()

    logging = callbacks.TensorBoard(
        log_dir=path.join('logs', cfg.exp_name),
        update_freq=100,
    )
    checkpointing = callbacks.ModelCheckpoint(
        cfg.save_to,
        verbose=1,
        save_weights_only=True,
    )

    net.fit_generator(
        dataset_train,
        epochs=100,
        callbacks=[logging, checkpointing],
        validation_data=dataset_val,
        validation_freq=1,
        max_queue_size=256,
        workers=12,
        use_multiprocessing=True,
    )

if __name__ == '__main__':
    main()

