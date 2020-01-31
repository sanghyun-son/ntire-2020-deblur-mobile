# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
import glob
import math
import itertools

import time
import argparse
import data
import metric

import numpy as np
import imageio
import tqdm

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='REDS')
    parser.add_argument('-m', '--model_file', type=str, default='models/deblur.tflite')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--save_results', action='store_true')
    parser.add_argument('-256', '--use_256', action='store_true')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(
        model_path=args.model_file,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    if args.test:
        input_dir = path.join(args.path, 'test', 'test_blur')
        target_dir = None
        save_dir = path.join('example', 'test')
    else:
        input_dir = path.join(args.path, 'val', 'val_blur')
        target_dir = path.join(args.path, 'val', 'val_sharp')
        save_dir = path.join('example', 'val')

    if args.save_results:
        os.makedirs(save_dir, exist_ok=True)

    scan = lambda x, y: glob.glob(path.join(x, y, '*.png'))
    input_list = [scan(input_dir, d) for d in os.listdir(input_dir)]
    input_list = sorted([it for it in itertools.chain(*input_list)])
    if target_dir is None:
        target_list = [None for _ in input_list]
    else:
        target_list = [scan(target_dir, d) for d in os.listdir(target_dir)]
        target_list = sorted([it for it in itertools.chain(*target_list)])

    # NxHxWxC, H:1, W:2
    psnr_avg = 0
    tq = tqdm.tqdm(zip(input_list, target_list), total=len(input_list))
    for input_path, target_path in tq:
        input_img = imageio.imread(input_path)
        if target_path is not None:
            target_img = imageio.imread(target_path)

        if floating_model:
            input_img = data.normalize(input_img)

        input_data = np.expand_dims(input_img, axis=0)
        if args.use_256:
            # Split the image into 256 x 256 patches
            input_list = []
            ps = 256
            _, h, w, _ = input_data.shape
            nh = math.ceil(h / ps)
            nw = math.ceil(w / ps)

            mh = (nh * ps - h) // (nh - 1)
            mw = (nw * ps - w) // (nw - 1)
            for ih in range(nh):
                ph = ih * (ps - mh)
                for iw in range(nw):
                    pw = iw * (ps - mw)
                    patch = input_data[..., ph:ph + ps, pw:pw + ps, :]
                    input_list.append(patch)
        else:
            input_list = [input_data]

        output_list = []
        for x in input_list:
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.squeeze(output_data)
            output_list.append(output_data)

        if args.use_256:
            cnt = 0
            row = []
            for ih in range(nh):
                col = []
                for iw in range(nw):
                    y = output_list[cnt]
                    if iw == 0:
                        col.append(y[:, :ps - mw // 2])
                    elif iw == nw - 1:
                        col.append(y[:, mw // 2:])
                    else:
                        col.append(y[:, mw // 2:ps - mw // 2])

                    cnt += 1

                col_cat = np.concatenate(col, axis=1)
                if ih == 0:
                    row.append(col_cat[:ps - mh // 2])
                elif ih == nh - 1:
                    row.append(col_cat[mh // 2:])
                else:
                    row.append(col_cat[mh // 2:ps - mh // 2])

            result = np.concatenate(row, axis=0)
        else:
            result = output_list[0]

        if floating_model:
            result = data.unnormalize(result)

        if target_path is not None:
            psnr_avg += metric.psnr_np(result, target_img)
            if args.save_results:
                save_as = input_path.replace(input_dir, save_dir)
                os.makedirs(path.dirname(save_as), exist_ok=True)
                imageio.imwrite(save_as, result)

    if target_path is not None:
        print('Avg. PSNR: {:.2f}'.format(psnr_avg / len(input_list)))

if __name__ == '__main__':
    main()

