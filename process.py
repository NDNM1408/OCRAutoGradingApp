# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image
import torch
from vietocr.predict import Predictor
from vietocr.model.vocab import Vocab
from vietocr.model.transformerocr import VietOCR

import tools.program
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.save_load import load_model
from ppocr.postprocess import build_post_process
from ppocr.modeling.architectures import build_model
from ppocr.data import create_operators, transform
import paddle
import json
import cv2


import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


config_rec = {'vocab': 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ-',
              'device': 'cpu',
              'seq_modeling': 'transformer',
              'transformer': {'d_model': 256,
                              'nhead': 8,
                              'num_encoder_layers': 6,
                              'num_decoder_layers': 6,
                              'dim_feedforward': 2048,
                              'max_seq_length': 1024,
                              'pos_dropout': 0.1,
                              'trans_dropout': 0.1},
              'optimizer': {'max_lr': 0.0005, 'pct_start': 0.1},
              'trainer': {'batch_size': 64,
                          'print_every': 200,
                          'valid_every': 3000,
                          'iters': 120000,
                          'export': './weights/transformerocr.pth',
                          'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
                          'log': './train.log',
                          'metrics': 10000},
              'dataset': {'name': 'hw1',
                          'data_root': '/kaggle/input/my-data/new_train/new_train/',  # forder chứa ảnh training
                          # forder chứa ảnh muốn dự đoán
                          'data_test_root': "/kaggle/input/my-data/public_test_data/new_public_test/",
                          'train_annotation': 'training_gt.txt',
                          'valid_annotation': 'valid_gt.txt',
                          'image_height': 32,
                          'image_min_width': 32,
                          'image_max_width': 256},
              'dataloader': {'num_workers': 3, 'pin_memory': True},
              'aug': {'data_aug': True, 'masked_language_model': True},
              'predictor': {'beamsearch': False},
              'quiet': False,
              'pretrain': '',
              'weights': 'final1.pth',
              'backbone': 'vgg19_bn',
              'cnn': {'pretrained': True,
                      'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                      'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                      'hidden': 256},
              'create_data_set': True}

predictor = Predictor(config=config_rec)


def sort_box(dt_boxes):
    sorted_boxes = sorted(dt_boxes, key=lambda box: (
        min(point[1] for point in box), min(point[0] for point in box)))
    max_height = 0
    for box in sorted_boxes:
        # print(box)
        local_max = max(abs(box[0][1] - box[3][1]), abs(box[1][1] - box[2][1]))
        max_height = max(max_height, local_max)
    thresh_hold = 0.2 * max_height
    horizontal_list = []
    box_list = []
    box_list.append(sorted_boxes[0])
    for box in sorted_boxes[1:]:
        box_prev = box_list[-1]
        prev = (box_prev[0][1] + box_prev[1][1] +
                box_prev[2][1] + box_prev[3][1]) / 4
        current = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
        if abs(current - prev) < thresh_hold:
            box_list.append(box)
        else:
            box_list = sorted(box_list, key=lambda box:
                              min(point[0] for point in box))
            horizontal_list.append(box_list)
            box_list = []
            box_list.append(box)
    box_list = sorted(box_list, key=lambda box:
                      min(point[0] for point in box))
    horizontal_list.append(box_list)
    return horizontal_list


def cut_box_and_save(box, img, img_name, save_path, predictor=predictor):
    # Extract the coordinates of the box
    x_coords, y_coords = zip(*box)
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Cut the box from the original image
    cut_box = img[y_min:y_max, x_min:x_max]

    img = Image.fromarray(cut_box)
    res = predictor.predict(img)
    return res


def draw_det_res(dt_boxes, config, img, img_name, save_path):
    # Sort the bounding boxes by their top-left coordinates (left to right, top to bottom)
    # sorted_boxes = sorted(dt_boxes, key=lambda box: (
    #     min(point[1] for point in box), min(point[0] for point in box)))

    # sorted_boxes = sorted(np.float32(dt_boxes), key=lambda c: (
    #     cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    sorted_box = sort_box(dt_boxes=dt_boxes)
    text = ""
    for line in sorted_box:
        for box_index, box in enumerate(line):
            res = cut_box_and_save(
                box, img, img_name, save_path)
            text = text + res + " "
        text = text + '\n'
    print(text)

    # logger.info(
    #     f"{len(dt_boxes)} bounding boxes sorted, cut, and saved as separate images in the following paths: {', '.join(saved_paths)}")
    logger.info("Successful")


@paddle.no_grad()
def main():
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                print
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json['points'] = np.array(box).tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(config['Global'][
                        'save_res_path']) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = np.array(box).tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(config['Global'][
                    'save_res_path']) + "/det_results/"
                draw_det_res(boxes, config, src_img, file, save_det_path)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())

    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = tools.program.preprocess()
    main()
