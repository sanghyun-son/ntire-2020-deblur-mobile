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
import time
import argparse
import data

import numpy as np
import imageio

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default='example/input.png')
    parser.add_argument('-m', '--model_file', type=str, default='models/deblur.tflite')
    parser.add_argument('-q', '--quantized', action='store_true')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor

    # NxHxWxC, H:1, W:2
    img = imageio.imread(args.image)

    save_dir = 'example'
    os.makedirs(save_dir, exist_ok=True)

    if args.quantized:
        img = img.astype(np.float32)
        img = img - 128
    else:
        img = data.normalize(img)

    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Note that we only measure the invoke time
    time_begin = time.time()
    interpreter.invoke()
    time_end = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    results = results.astype(np.float32)

    if args.quantized:
        results = results + 128
    else:
        results = 127.5 * (results + 1)

    results = results.round().clip(min=0, max=255)
    results = results.astype(np.uint8)

    imageio.imwrite(path.join(save_dir, 'output.png'), results)
    time_total = time_end - time_begin
    print('Time: {:.3f}s {:.1f}fps'.format(time_total, 1 / time_total))

if __name__ == '__main__':
    main()

