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
import cv2
import matplotlib.cm as mpcm

from dataset.coco import dataset_common

def gain_translate_table():
    label2name_table = {}
    for class_name, labels_pair in dataset_common.COCO_LABELS.items():
        label2name_table[labels_pair[0]] = class_name
    return label2name_table

label2name_table = gain_translate_table()
#
# def colors_subselect(colors, num_classes=81):
#     dt = len(colors) // num_classes
#     sub_colors = []
#     for i in range(num_classes):
#         color = colors[i*dt]
#         if isinstance(color[0], float):
#             sub_colors.append([int(c * 255) for c in color])
#         else:
#             sub_colors.append([c for c in color])
#     return sub_colors
#
# colors = colors_subselect(mpcm.plasma.colors, num_classes=81)

colors_tableau = [(255, 255, 255), (21, 6, 138), (29, 6, 141), (35, 5, 144), (41, 5, 147), (47, 4, 149),
                  (52, 4, 152), (58, 4, 154), (63, 3, 156), (68, 3, 158), (73, 2, 159), (78, 2, 161),
                  (82, 1, 163), (87, 1, 164), (92, 0, 165), (97, 0, 166), (101, 0, 167), (106, 0, 167),
                  (111, 0, 168), (115, 0, 168), (120, 1, 168), (124, 2, 167), (129, 4, 167), (133, 6, 166),
                  (137, 8, 165), (142, 12, 164), (146, 15, 162), (150, 18, 160), (154, 21, 158), (158, 25, 156),
                  (162, 28, 154), (165, 31, 151), (169, 35, 149), (173, 38, 146), (176, 42, 143), (180, 45, 141),
                  (183, 48, 138), (186, 52, 135), (189, 55, 132), (192, 59, 129), (195, 62, 127), (198, 65, 124),
                  (201, 69, 121), (204, 72, 118), (207, 75, 116), (209, 79, 113), (212, 82, 110), (215, 86, 108),
                  (217, 89, 105), (220, 93, 102), (222, 96, 100), (224, 100, 97), (227, 103, 95), (229, 107, 92),
                  (231, 110, 90), (233, 114, 87), (235, 118, 84), (237, 121, 82), (239, 125, 79), (240, 129, 77),
                  (242, 133, 74), (244, 137, 71), (245, 141, 69), (246, 145, 66), (248, 149, 64), (249, 153, 61),
                  (250, 157, 58), (251, 162, 56), (252, 166, 53), (252, 170, 51), (253, 175, 49), (253, 179, 46),
                  (253, 184, 44), (253, 188, 42), (253, 193, 40), (253, 198, 38), (252, 203, 37), (251, 208, 36),
                  (250, 213, 36), (249, 217, 36), (248, 223, 36)]

# colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        # if classes[i] < 1: continue
        bbox = bboxes[i]
        color = colors_tableau[classes[i]]
        # Draw bounding boxes
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text
        s = '%s/%.1f%%' % (label2name_table[classes[i]], scores[i]*100)
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img

