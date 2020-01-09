# NTIRE 2020 Image Deblurring Challenge: Track 2 on Smartphone

This repository provides a basic tutorial for the NTIRE 2020 Image Deblurring Challenge: Track 2. Please read the following guidelines carefully to deploy your model on the real smartphone device.

## Environment

We recommend the conda environment on the Linux machine. Please clone this repository and import the given conda environment by following:

```bash
git clone https://github.com/thstkdgus35/ntire-2020-deblur-mobile.git
cd ntire-2020-deblur-mobile
conda env create -f environment.yml
```

[TensorFlow 2.0](https://www.tensorflow.org/) and [PyTorch 1.3.1](https://pytorch.org/) will be installed by default. You can use any libraries, but we heavily recommend the TensorFlow for easier mobile deployment.

We also note that this repository is verified in the following environments:
* Ubuntu 16.04
* CUDA 10.0 (For TensorFlow) / CUDA 10.2 (For PyTorch)

## Prepare the dataset

First, download the **REDS_deblur** dataset from the following links.

* Training data
  * Blur: [Google drive](https://drive.google.com/open?id=1Be2cgzuuXibcqAuJekDgvHq4MLYkCgR8) / [SNU CVLab](https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/train_blur.zip)
  * Sharp: [Google drive](https://drive.google.com/open?id=1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-) / [SNU CVLab](https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/train_sharp.zip)

* Validation data
  * Blur: [Google drive](https://drive.google.com/open?id=1N8z2yD0GDWmh6U4d4EADERtcUgDzGrHx) / [SNU CVLab](https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/val_blur.zip)
  * Sharp: [Google drive](https://drive.google.com/open?id=1MGeObVQ1-Z29f-myDP7-8c3u0_xECKXq) / [SNU CVLab](https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/val_sharp.zip)

* Test data
  * Blur: [Google drive](https://drive.google.com/file/d/1dr0--ZBKqr4P1M8lek6JKD1Vd6bhhrZT/view?usp=sharing) / [SNU CVLab](https://cv.snu.ac.kr/~snah/Deblur/dataset/REDS/test_blur.zip)

Your data should be organized as following:

```bash
$(THIS_REPOSITORY)
|--REDS_deblur
|   |-- train
|   |   |-- train_blur
|   |   |   |-- 000
|   |   |   |-- 001
|   |   |   `-- ...
|   |   `-- train_sharp
|   |       `-- ...
|   |-- val
|   |   |-- val_blur
|   |   |   `-- ...
|   |   `-- val_sharp
|   |       `-- ...
|   `-- test
|       `-- test_blur
|           `-- ...
|-- README.md
|-- main.py
|-- preprocess.py
`-- ...
```

Since images in the dataset are pretty large (1280 x 720), we recommend to preprocessing to save time for data loading. For example, you can run the `preprocess.py` to crop each frame of REDS_deblur dataset into 16 subregions.

```bash
# You are in $(THIS_REPOSITORY)/.
# This may take some time...
$ python preprocess.py
```

After then, you will get `train_crop` under `REDS_deblur`. We note that this preprocessing **does not** affect to the number of effective training patches.


## Training a baseline model

We provide a baseline code for those who are not familiar with TensorFlow. Below is a list of files related to the training stage.

```bash
main.py
data.py
model.py
metric.py
```

You can start the training by the following:

```bash
# You are in $(THIS_REPOSITORY)/.
$ python main.py --exp_name [EXPERIMENT_NAME]

Additional arguments:
    --patch_size: Training patch size
    --batch_size: Training batch size
    --epochs    : The number of total epochs
    --save_to   : Path of the model checkpoint (weights ONLY) to be saved
```

Training logs will be saved under `logs/[EXPERIMENT_NAME]`. Find them out with TensorBoard:
```bash
# You are in $(THIS_REPOSITORY)/.
$ tensorboard --logdir logs
```

By default, [`localhost:6006`](http://localhost:6006/) will show you training and evaluation curves.


## Convert your model to a TFLite model

If you are training your model with the TensorFlow framework, it is straightforward to convert them into a TFLite model.

```bash
# You are in $(THIS_REPOSITORY)/.
$ python convert_model.py

Additional arguments:
    --img_h    : Height of the test image(s) (should be FIXED)
    --img_w    : Width of the test image(s) (should be FIXED)
    --load_from: Path to the model checkpoint
    --save_to  : Path of the TFLite model to be saved
```

If you want to convert your PyTorch model to a TFLite model, please follow the guideline below.


## Test your TFLite model

You can easily check the converted TFLite model on your PC.

```bash
# You are in $(THIS_REPOSITORY)/.
$ python test_deblur.py

Additional arguments:
    -i, --image     : Path to the input image
    -m, --model_file: Path to the TFLite model
```

You can find a result image from `example/output.png`.


## Test your TFLite model on virtual Android environment

Will be prepared soon...


## Test your TFLite model on a real Android device

Will be prepared soon...


## PyTorch `state_dict` to a TFLite model

Unfortunately, there is no straightforward way to convert your PyTorch `state_dict` to a TFLite model directly. The major problem comes from the channel convention: while `(N, C, H, W)` is common in PyTorch, TFLite only supports `(N, H, W, C)`. Please follow the steps below carefully to transfer your model from PyTorch.

1) Train your model on PyTorch.
2) Save your PyTorch `state_dict` (i.e., `torch.save(model.state_dict(), 'state_dict.pth')`).
3) Define a [Keras model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) which has the same structure to your PyTorch model.
4) Will be prepared soon...