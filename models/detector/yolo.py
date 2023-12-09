#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @file: yolo.py
# @author: jerrzzy
# @date: 2023/7/13


from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
from .nets.yolo import YoloBody
from .utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input, resize_image, show_config)
from .utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {
        "model_path"        : './models/detector/model_data/yolov4_tiny_wheelworm_weights.pth',
        "classes_path"      : './models/detector/model_data/worm_classes.txt',
        "anchors_path"      : './models/detector/model_data/yolo_anchors.txt',
        "anchors_mask"      : [[3, 4, 5], [1, 2, 3]],
        "phi"               : 0,
        "input_shape"       : [416, 416],
        "confidence"        : 0.35,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        self.generate()
        show_config(**self._defaults)

    def generate(self):
        """生成模型"""
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect(self, image):
        """检测单张图片"""
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
            if results[0] is None:
                return None
            boxes = results[0][:, :4]
        return boxes

