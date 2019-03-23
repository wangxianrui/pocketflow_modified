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
import shutil
import tensorflow as tf

from pycocotools.coco import COCO
from utils.external.ssd_tensorflow.dataset.coco.pycocotools.cocoeval import COCOeval
from utils.external.ssd_tensorflow.dataset.coco.dataset_common import COCO_LABELS
from utils.external.ssd_tensorflow.dataset.coco.dataset_common import labelmap


def do_python_eval(dataset_path, pred_path):
    output_path = os.path.join(pred_path, 'eval_output')
    anno_files = os.path.join(dataset_path, 'annotations', 'minival.json')

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # extract data
    print('extracting data from txt...')
    data_list = []
    for cls_name, cls_pair in COCO_LABELS.items():
        if 'none' in cls_name:
            continue
        cls_id = cls_pair[0]
        with open(os.path.join(pred_path, 'results_%d.txt' % cls_id), 'r') as file:
            # [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},
            # print(file)
            for line in file.readlines():
                line = line.split(' ')
                if len(line) != 0:
                    data = dict()
                    data['image_id'] = int(line[0])
                    # convert to original label
                    for k in labelmap.keys():
                        if labelmap[k] == int(line[1]):
                            data['category_id'] = int(k)
                    data['bbox'] = [float(line[2]), float(line[3]), float(line[4]), float(line[5])]
                    data['score'] = float(line[6])
                    data_list.append(data)

    # filter predictions
    print('filtering predictions...')
    id_set = set()
    for pred in data_list:
        id_set.add(pred['image_id'])

    dic = dict()
    for id in id_set:
        dic[id] = []

    for pred in data_list:
        dic[pred['image_id']].append(pred)

    final_list = []
    for id in id_set:
        dic[id].sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(dic[id]) >= 200:
            dic[id] = dic[id][:200]
        for item in dic[id]:
            final_list.append(item)

    with open(os.path.join(output_path, 'results.json'), 'w') as file:
        file.write(str(final_list).replace("'", '"'))

    # do cocoeval
    print('coco evaluating...')
    filename = os.path.join(output_path, 'results.json')
    # load GT
    cocoGt = COCO(anno_files)
    cocoDt = cocoGt.loadRes(filename)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    # mAP = cocoEval.summarize()
    # tf.summary.scalar('mAP', mAP)
    cocoEval.summarize()


if __name__ == '__main__':
        do_python_eval('/DATA/coco', '/home/hzh/hzh/wxrui/PocketFlow/ssd_outputs')
