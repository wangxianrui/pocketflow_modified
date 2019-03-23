
##VOC dataset

### DATASET:
python utils/external/ssd_tensorflow/dataset/convert_tfrecords_VOC.py \
--dataset_directory /home/wxrui/DATA/VOCdevkit


### VGG VOC:
- python nets/vgg_at_pascalvoc_run.py \
--image_size 300 \
--data_dir_local /home/wxrui/DATA/VOCdevkit

- python nets/vgg_at_pascalvoc_run.py \
--image_size 300 \
--data_dir_local /home/wxrui/DATA/VOCdevkit \
--exec_mode eval

- python nets/vgg_at_pascalvoc_run.py \
--image_size 300 \
--data_dir_local /home/wxrui/DATA/VOCdevkit \
--learner uniform-tf

- python nets/vgg_at_pascalvoc_run.py \
--image_size 300 \
--data_dir_local /home/wxrui/DATA/VOCdevkit \
--learner uniform-tf \
--exec_mode eval


### PELEE VOC:

- python nets/peleenet_at_pascalvoc_run.py \
--image_size 304 \
--data_dir_local /home/wxrui/DATA/VOCdevkit

- python nets/peleenet_at_pascalvoc_run.py \
--image_size 304 \
--data_dir_local /home/hzh/DATA/VOCdevkit \
--exec_mode eval

- python nets/peleenet_at_pascalvoc_run.py \
--image_size 304 \
--data_dir_local /home/wxrui/DATA/VOCdevkit \
--learner uniform-tf

- python nets/peleenet_at_pascalvoc_run.py \
--image_size 304 \
--data_dir_local /home/wxrui/DATA/VOCdevkit \
--learner uniform-tf \
--exec_mode eval




## COCO dataset

