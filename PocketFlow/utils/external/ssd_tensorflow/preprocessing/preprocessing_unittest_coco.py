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
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np
import sys; sys.path.insert(0, ".")
from utility import draw_toolbox_coco
import ssd_preprocessing

slim = tf.contrib.slim

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image)

    img_to_draw = draw_toolbox_coco.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
    imsave(os.path.join('./tmp_coco/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter

def slim_get_split(file_pattern='{}_????'):
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/image_id': tf.FixedLenFeature([1], tf.int64),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'image_id': slim.tfexample_decoder.Tensor('image/image_id'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=100,
                items_to_descriptions=None,
                num_classes=81,
                labels_to_names=None)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=2,
                    common_queue_capacity=32,
                    common_queue_min=8,
                    shuffle=True,
                    num_epochs=1)

    [org_image, image_id, shape, glabels_raw, gbboxes_raw] = provider.get(['image', 'image_id', 'shape',
                                                                         'object/label',
                                                                         'object/bbox'])
    image, glabels, gbboxes = ssd_preprocessing.preprocess_image(org_image, glabels_raw, gbboxes_raw, [304, 304], is_training=True, data_format='channels_last', output_rgb=False)

    # image = tf.transpose(image, perm=(1, 2, 0))
    save_image_op = tf.py_func(save_image_with_bbox,
                            [ssd_preprocessing.unwhiten_image(image),
                            tf.clip_by_value(glabels, 0, tf.int64.max),
                            tf.ones_like(glabels),
                            gbboxes],
                            tf.int64, stateful=True)

    # out_shape = [304, 304]
    # image = tf.to_float(org_image)
    # image = tf.image.resize_images(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    # image.set_shape(out_shape + [3])

    # save_image_op = tf.py_func(save_image_with_bbox, [image, glabels_raw, tf.ones_like(glabels_raw), gbboxes_raw], tf.int64, stateful=True)

    """
        ============before ssd_preprocessing===============: 
        image: Tensor("dataset_data_provider/case/cond/Merge:0", shape=(?, ?, 3), dtype=uint8) 
        glabels_raw: Tensor("dataset_data_provider/SparseToDense:0", shape=(?,), dtype=int64) 
        gbboxes_raw: Tensor("dataset_data_provider/transpose:0", shape=(?, 4), dtype=float32)
        
        ============after ssd_preprocessing================: 
        image: Tensor("ssd_preprocessing_train/merge_bgr:0", shape=(304, 304, 3), dtype=float32) 
        glabels_raw: Tensor("ssd_preprocessing_train/ssd_random_sample_patch_wrapper/cond/Merge_1:0", shape=(?,), dtype=int64) 
        gbboxes_raw: Tensor("ssd_preprocessing_train/random_flip_left_right/cond_1/Merge:0", shape=(?, 4), dtype=float32)
    
    """

    return save_image_op

if __name__ == '__main__':
    save_image_op = slim_get_split('/DATA/coco/tfrecords/*')
    # Create the graph, etc.
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            print(sess.run(save_image_op))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
