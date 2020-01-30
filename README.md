# NTIRE 2020 Image Deblurring Challenge: Track 2 on Smartphone

This repository provides a basic tutorial for the NTIRE 2020 Image Deblurring Challenge: Track 2. Please read the following guidelines carefully to deploy your model on the real smartphone device.

## Environment

We recommend the conda environment on the Linux machine. Please clone this repository and import the given conda environment by following:

```bash
git clone https://github.com/thstkdgus35/ntire-2020-deblur-mobile.git
cd ntire-2020-deblur-mobile
conda env create -f environment.yml
```

[TensorFlow 2.1](https://www.tensorflow.org/) and [PyTorch 1.4](https://pytorch.org/) will be installed by default. You can use any libraries of any versions, but we heavily recommend the TensorFlow for easier mobile deployment. We note that the final goal of this challenge track is to submit `.tflite` model.

This repository is verified in the following environments:
* Ubuntu 16.04 / 18.04
* CUDA 10.0 (For TensorFlow) / CUDA 10.1 (For PyTorch)
* CuDNN 7.6.5

## Prepare the dataset

First, download the **REDS** dataset from the following links.

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
ntire-2020-deblur-mobile
|--REDS
|   |-- train
|   |   |-- train_blur
|   |   |   |-- 000
|   |   |   |-- 001
|   |   |   `-- ...
|   |   `-- train_sharp
|   |   `-- ...
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

Since images in the dataset are pretty large (1280 x 720), we recommend a preprocessing stage to save time for data loading. You can run the `preprocess.py` to crop each frame of the `REDS` dataset into 16 subregions.

```bash
# You are in ntire-2020-deblur-mobile/.
# This may take some time...
$ python preprocess.py
```

After then, you will get `train_crop` under `REDS_deblur`. We note that this **does not** affect to the number of effective training patches.


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
# You are in ntire-2020-deblur-mobile/.
$ python main.py --exp_name [EXPERIMENT_NAME]

Additional arguments:
    --patch_size: Training patch size
    --batch_size: Training batch size
    --epochs    : The number of total epochs
    --lr        : Initial learning rate
    --lr_gamma  : Learning rate decay factor
    --milestones: Learning rate schedule (ex: --milestones 10 20 30)
    --save_to   : Path of the model checkpoint (weights ONLY) to be saved
```

Training logs will be saved under `logs/[EXPERIMENT_NAME]`. Find them out with TensorBoard:
```bash
# You are in ntire-2020-deblur-mobile/.
$ tensorboard --logdir logs
```

By default, [`localhost:6006`](http://localhost:6006/) will show you training and evaluation curves.


## Convert your model to a `.tflite` model

If you have trained your model with the TensorFlow framework, it is straightforward to convert them into a `.tflite` model.

```bash
# You are in ntire-2020-deblur-mobile/.
# Model for evaluating PSNR
$ python convert_model.py
# Model for evaluating timing
$ python convert_model.py --save_to deblur_256.tflite --test example/input_256.png

Additional arguments:
    -l, --load_from : Path to the model checkpoint
    -s, --save_to   : Path to the .tflite model to be saved
    -t, --test      : Path to the input image to be fed
```

We note that this step is dependent on **image resolution**.
Therefore, different input image sizes may result in different `.tflite` models.
If you want to convert your PyTorch model to a `.tflite` model, please follow the guideline below.


## Test your `.tflite` model

You can easily check and evaluate the converted `.tflite` model on your PC.

```bash
# You are in ntire-2020-deblur-mobile/.
$ python test_deblur.py
# or
# Check the execution on a 256x256 input.
$ python test_deblur.py -i example/input_256.png -m models/deblur_256.tflite
# or
# Test the model on the val split
$ python test_deblur_full.py -m models/deblur.tflite
# Generate result images on the test split
$ python test_deblur_full.py -m models/deblur.tflite -t -s

Additional arguments for test_deblur.py:
    -i, --image         : Path to the input image
    -m, --model_file    : Path to the .tflite model

    We note that the .tflite model is dependent to input image resolution.

Additional arguments for test_deblur_full.py:
    -m, --model_file    : Path to the .tflite model (for 1280x720 inputs)
    -t, --test          : Use the test split
    -s, --save_results  : Save result images under example/val or example/test
```

You can find a result image from `example/output.png`.
We note that this step may not support GPU acceleration and take several hours in the case.


## Test your `.tflite` model on a real Android device

We note that final submissions should be generated from the submitted `.tflite` model.
We will double-check whether submitted deblurred images can be acquired from the submitted `.tflite` models.
However, for easier performance evaluation, we will execute the `.tflite` model on a real Android device **only** for measuring runtime.
Detailed comments on this concept can be found from [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

In short, we will measure the average runtime of each model using randomly generated inputs.
As a result, input-dependent models will not be allowed at this moment (and it is very challenging to optimize those input-dependent models for mobile devices).

We use the [Google Pixel 4](https://store.google.com/?srp=/product/pixel_4) device (Android 10) for evaluation.
If you are going to use different devices, please change some API versions to appropriate ones.
More detailed explanations of the tutorial can be found from [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android), but we recommend to follow package versions of the configurations below.

First, install the [Android Studio](https://developer.android.com/studio/install?hl=en) and SDK by running the studio.
[Android NDK](https://developer.android.com/ndk/guides) is also required, and we note that the NDK version (**18b**) matters.
Please download the zip file from [here](https://developer.android.com/ndk/downloads/older_releases.html#ndk-18b-downloads) and unzip it.
If you are not sure about the file path, you can skip this now.

Then, clone the TensorFlow official repository by following:

```bash
# You can run the below scripts from anywhere you want.
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
```

Bazel will be used to build the evaluation binary.
Detailed descriptions can be found from [here](https://docs.bazel.build/versions/master/install-ubuntu.html).
We note that the Bazel version (**1.2.1**) matters.

```bash
# You can run the below scripts from anywhere.
$ sudo apt install curl
$ curl https://bazel.build/bazel-release.pub.gpg \
    | sudo apt-key add -
$ echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | sudo tee /etc/apt/sources.list.d/bazel.list
$ sudo apt update
# Please specifiy the Bazel version.
$ sudo apt install bazel-1.2.1
```

Before building the evaluation binary, please set the configuration by following:

```bash
# You are in tensorflow/.
$ ./configure
Please specify the location of python. [ENTER]
Please input the desired Python library path to use. [ENTER]
Do you wish to build TensorFlow with XLA JIT support? [n]
Do you wish to build TensorFlow with OpenCL SYCL support? [N]
Do you wish to build TensorFlow with ROCm support? [N]
Do you wish to build TensorFlow with CUDA support? [y]
Do you wish to build TensorFlow with TensorRT support? [N]
Please specify a list of comma-separated CUDA compute capabilities you want to build with. (...) [ENTER]
Do you want to use clang as CUDA compiler? [N]
Please specify which gcc should be used by nvcc as the host compiler. [ENTER]
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified. [ENTER]
Would you like to interactively configure ./WORKSPACE for Android builds? [y]
# Now you require the NDK from above.
# $(PATH_TO_NDK_DIRECTORY)
# |-- build
# |-- CHANGELOG.md
# |-- meta
# |-- ...
# `-- wrap.sh
Please specify the home path of the Android NDK to use. [PATH_TO_NDK_DIRECTORY]
Please specify the (min) Android NDK API level to use. [21]
Please specify the home path of the Android SDK to use. [ENTER]
Please specify the Android SDK API level to use. [29]
Please specify an Android build tools version to use. [29.0.2]
```

Then, build the evaluation binary.
We note that we are not building the whole TensorFlow.

```bash
# You are in tensorflow/.
$ bazel build -c opt \
    --config=android_arm \
    --cxxopt='--std=c++14' \
    tensorflow/lite/tools/benchmark:benchmark_model
```

After the build successes, connect your Android device to PC.
Please make sure that you have enabled the [USB debugging mode](https://developer.android.com/studio/debug/dev-options?hl=en) and allowed file transfer from PC.
Now you are ready to run the `.tflite` model on a real Android device.

```bash
# You are in tensorflow/.
$ adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
$ adb shell chmod +x /data/local/tmp/benchmark_model
$ adb push [PATH_TO_TFLITE_MODEL]/[NAME_OF_TFLITE_MODEL] /data/local/tmp
$ adb shell /data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/[NAME_OF_TFLITE_MODEL] \
    --num_threads=4
```

Please find out **Average inference timings in us** from the printed logs.
We note that Wrapup & Init timings will not be considered in this challenge.

We report performances of the provided baseline model.
For some hardware-related issues, timings is measured with inputs of 256 x 256 (`models/deblur_256.tflite`) while 1280 x 720 inputs (`models/deblur.tflite`) are used to measure PSNR.
More options can be found from [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).
Detailed analysis of baseline models and their quantized version will be uploaded soon.

| Avg. timing(ms) / FPS (50 runs) | CPU | CPU | CPU | PSNR(dB)
|:--------------:|:-----------:|:----------:|:----------:|:----:|
| FP32 and FP16 | 1762 / 0.57 | 1146 / 0.87 | 775 / 1.30 | 28.34 |
| Quantized | - | - | - | - |
| | `--num_threads=1` | `--num_threads=2` | `--num_threads=4` | |

| Avg. timing(ms) / FPS (50 runs) | CPU | GPU | NNAPI | PSNR
|:--------------:|:----------:|:----------:|:----------:|:----:|
| FP32 and FP16 | 768 / 1.30 | 121 / 8.23 | 226 / 4.42 | 28.34 |
| Quantized | - | - | - | - |
| | `--num_threads=8` | `--use_gpu=true` | `--use_nnapi=true` | |

* Notes
  * While FP16 models (e.g., `models/deblur_fp16_256.tflite`) are smaller than FP32 models, they may not have advantages compared to FP32 counterparts in terms of runtime. Please check the [link](https://www.tensorflow.org/lite/performance/post_training_float16_quant).

## PyTorch `state_dict` to a `.tflite`  model

Unfortunately, there is no straightforward way to convert your PyTorch `state_dict` to a `.tflite` model directly.
The major problem comes from the channel convention: while `(N, C, H, W)` is a standard in PyTorch, `.tflite` only supports `(N, H, W, C)`.
Please follow the steps below carefully to transfer your model from PyTorch.

1) Train your model on PyTorch.
2) Save your PyTorch `state_dict` (i.e., `torch.save(model.state_dict(), 'state_dict.pth')`).
3) Define a [Keras model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) which has the same structure to your PyTorch model. You can compare `model_pth.py` and `model.py` to see differences of two frameworks.
4) Will be prepared soon...
