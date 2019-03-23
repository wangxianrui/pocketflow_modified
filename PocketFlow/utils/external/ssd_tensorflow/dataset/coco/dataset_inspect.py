# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import IPython.display as display

'''
def count_split_examples(split_path, file_prefix='.tfrecord'):
    # Count the total number of examples in all of these shard
    num_samples = 0
    tfrecords_to_count = tf.gfile.Glob(os.path.join(split_path, file_prefix))
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):#, options = opts):
            num_samples += 1
    return num_samples
'''


# count the example in tfrecords
def count_example(dir, pattern):
    file_list = tf.gfile.Glob(os.path.join(dir, pattern))
    example = 0
    for file in file_list:
        for record in tf.io.tf_record_iterator(file):
            example += 1
    return example


# get the data structure of one sample
def get_structure(dir, filename):
    record_iterator = tf.python_io.tf_record_iterator(os.path.join(dir, filename))

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # show image only in jupyter notebook
        # image_raw = example.features.feature['image/encoded'].bytes_list.value[0]
        # display.display(display.Image(data=image_raw))

        with open('data.txt', 'w') as file:
            file.write(str(example))


# check whether there are iamges without bboxes
def inspect_bbox(dir, pattern):
    file_list = tf.gfile.Glob(os.path.join(dir, pattern))
    i = 0
    count = 0
    for file in file_list:
        for string_record in tf.python_io.tf_record_iterator(file):
            example = tf.train.Example()
            example.ParseFromString(string_record)

            i += 1
            image_id = example.features.feature['image/image_id'].int64_list.value

            xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
            # xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
            # ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
            # ymin = example.features.feature['image/object/bbox/ymin'].float_list.value
            # label = example.features.feature['image/object/bbox/label']
            # category = example.features.feature['image/object/bbox/label'].int64_list.value

            if xmax == []:
                # print(image_id)
                count += 1
        print(file, 'has been checked.')

    print(i, 'images have been checked.')
    print(count, 'annotations without bbox.')


if __name__ == '__main__':
    print('train:', count_example('/DATA/coco/tfrecords', 'train-?????-of-?????'))
    print('val:', count_example('/DATA/coco/tfrecords', 'val-?????-of-?????'))
    get_structure('/DATA/coco/tfrecords', 'train-00000-of-00016')
    inspect_bbox('/DATA/coco/tfrecords', 'train-?????-of-?????')
    inspect_bbox('/DATA/coco/tfrecords', 'val-?????-of-?????')
