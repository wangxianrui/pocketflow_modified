from abc import ABC
import tensorflow as tf
from datasets.coco_dataset import COCODataset
# from datasets.pascalvoc_dataset import PascalVocDataset

# def build(self, enbl_trn_val_split=False):
#     '''Build iterator(s) for tf.data.Dataset() object.
#
#     Args:
#     * enbl_trn_val_split: whether to split into training & validation subsets
#
#     Returns:
#     * iterator_trn: iterator for the training subset
#     * iterator_val: iterator for the validation subset
#       OR
#     * iterator: iterator for the chosen subset (training OR testing)
#
#     Example:
#       # build iterator(s)
#       dataset = xxxxDataset(is_train=True)  # TF operations are not created
#       iterator = dataset.build()            # TF operations are created
#           OR
#       iterator_trn, iterator_val = dataset.build(enbl_trn_val_split=True)  # for dataset-train only
#
#       # use the iterator to obtain a mini-batch of images & labels
#       images, labels = iterator.get_next()
#     '''
#
#     # obtain list of data files' names
#     filenames = tf.data.Dataset.list_files(self.file_pattern, shuffle=True)
#
#
#     # create a tf.data.Dataset from list of files
#     dataset = filenames.apply(
#         tf.contrib.data.parallel_interleave(self.dataset_fn, cycle_length=4))
#     dataset = dataset.map(self.parse_fn, num_parallel_calls=8)
#
#     # create iterators for training & validation subsets separately
#     return self.__make_iterator(dataset)
#
#
# def __make_iterator(self, dataset):
#     """Make an iterator from tf.data.Dataset.
#
#     Args:
#     * dataset: tf.data.Dataset object
#
#     Returns:
#     * iterator: iterator for the dataset
#     """
#
#     dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024))
#     dataset = dataset.batch(32)
#     dataset = dataset.prefetch(16)
#     iterator = dataset.make_one_shot_iterator()
#
#     return iterator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('enbl_multi_gpu', False, 'enable multi-GPU training')

iterator = COCODataset(is_train=True).build()
# iterator = PascalVocDataset(is_train=True).build()
feature = iterator.get_next()

"""
    structure of feature:
    (
        {
            'image': array([[[[]]]], dtype=float32), # (N, W, H, C): (32, 304, 304, 3)
            'image_id': array([[]]), # (N, 1)
            'shape': array([[]]) # (N, 3)
        },
        
        array([[[]]], dtype=float32) # (N, 93, 6)  '6' represent: (1 or 0, x,x,x,x, label)
    )
"""


with tf.Session() as sess:
    # for i in range(1):
    #     print('\n=======from coco_dataset.py: LOOP', i + 1, '=======\n', sess.run(feature))
    tuple = sess.run(feature)
    print(tuple[0]['image'].shape)
    print(tuple[0]['image_id'].shape)
    print(tuple[0]['shape'].shape)
    print(tuple[1].shape)
