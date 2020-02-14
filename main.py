import os
from os import path
import argparse
import bisect

import tensorflow as tf
from tensorflow import lite
from tensorflow.keras import callbacks

import model
import data
import metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--keep_range', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--milestones', nargs='+', default=[10, 15])
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--save_as', type=str, default='models/deblur.hdf5')
    cfg = parser.parse_args()

    # For checking the GPU usage
    #tf.debugging.set_log_device_placement(True)
    # For limiting the GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    dataset_train = data.REDS(
        cfg.batch_size,
        patch_size=cfg.patch_size,
        train=True,
        keep_range=cfg.keep_range,
    )
    # We have 3,000 validation frames.
    # Note that each frame will be center-cropped for the validation.
    dataset_val = data.REDS(
        20,
        patch_size=cfg.patch_size,
        train=False,
        keep_range=cfg.keep_range,
    )

    if cfg.depth == 4:
        net = model.Baseline(cfg.patch_size, cfg.patch_size)
    else:
        net_class = getattr(model, 'Small{}'.format(cfg.depth))
        net = net_class(cfg.patch_size, cfg.patch_size)

    net.build(input_shape=(None, cfg.patch_size, cfg.patch_size, 3))
    kwargs = {'optimizer': 'adam', 'loss': 'mse'}
    if cfg.keep_range:
        net.compile(**kwargs, metrics=[metric.psnr_full])
    else:
        net.compile(**kwargs, metrics=[metric.psnr])
    net.summary()

    # Callback functions
    # For TensorBoard logging
    logging = callbacks.TensorBoard(
        log_dir=path.join('logs', cfg.exp_name),
        update_freq=100,
    )
    # For checkpointing
    os.makedirs(path.dirname(cfg.save_as), exist_ok=True)
    checkpointing = callbacks.ModelCheckpoint(
        cfg.save_as,
        verbose=1,
        save_weights_only=True,
    )
    def scheduler(epoch):
        idx = bisect.bisect_right(cfg.milestones, epoch)
        lr = cfg.lr * (cfg.lr_gamma**idx)
        return lr
    # For learning rate scheduling
    scheduling = callbacks.LearningRateScheduler(scheduler, verbose=1)

    net.fit_generator(
        dataset_train,
        epochs=cfg.epochs,
        callbacks=[logging, checkpointing, scheduling],
        validation_data=dataset_val,
        validation_freq=1,
        max_queue_size=16,
        workers=8,
        use_multiprocessing=True,
    )

if __name__ == '__main__':
    main()

