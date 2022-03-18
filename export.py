#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import print_function
import argparse
import os

import tensorflow as tf
from utils.helpers import dump_frozen_graph
from mnist import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export model in IE format')
    parser.add_argument('--model_name', default='mnist')
    parser.add_argument('--output_dir', default=None, help='Output Directory')
    parser.add_argument('--checkpoint', default=None, help='Default: latest')
    return parser.parse_args()

def freezing_graph(checkpoint, output_dir):

    with tf.Session() as sess:
        image = tf.placeholder(dtype=tf.float32,
                                      shape=(None, ) +
                                      tuple([28,28]),name='Placeholder')
        
        model = create_model('channels_last')
        tf.get_variable('global_step',
                        initializer=tf.constant_initializer(0, dtype=tf.int32),
                        shape=(),
                        dtype=tf.int32,
                        trainable=False)
        logits = model(image,training=False)
        prediction = tf.nn.softmax(logits,name='Softmax')
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        tf.saved_model.simple_save(sess,output_dir,inputs={'input':image},
                                           outputs={'prediction':prediction})
        return output_dir


def main(_):
    args = parse_args()

    checkpoint = args.checkpoint
    print(checkpoint)
    if not checkpoint or not os.path.isfile(checkpoint + '.index'):
        raise FileNotFoundError(str(checkpoint))

    step = checkpoint.split('-')[-1]
    output_dir = args.output_dir 

    # Freezing graph
    frozen_dir = os.path.join(output_dir, 'frozen_graph')
    output_dir = freezing_graph(checkpoint, frozen_dir)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
