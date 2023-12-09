#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @folder: detector
# @author: jerrzzy
# @date: 2023/8/26


__all__ = ['mot_detect']


import cv2
from .yolo import YOLO
from tqdm import tqdm


def mot_detect(input_video_path, txt_result_path, total_frames, input_shape, cuda, confidence, nms_iou):
    """
    :param input_video_path: 输入视频路径
    :param txt_result_path: txt格式缓存结果保存路径
    :param cfg: 字典对象的配置文件
    :return: None
    """
    print('start detect...')
    detector = YOLO(
        input_shape=input_shape,
        cuda=cuda,
        confidence=confidence,
        nms_iou=nms_iou
    )
    cv2_video_obj = cv2.VideoCapture(input_video_path)
    frame_id = 1
    with open(txt_result_path, 'w') as f:
        pbar = tqdm(total=total_frames)
        while True:
            ret, frame = cv2_video_obj.read()
            if not ret:
                break
            bboxes = detector.detect(frame)
            if bboxes is not None:
                for (x1, y1, x2, y2) in bboxes:
                    """
                    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    """
                    f.write(f'%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame_id, -1, x1, y1, x2-x1, y2-y1)+'\n')
            pbar.update(1)
            frame_id += 1
    f.close()
    cv2.destroyAllWindows()
    cv2_video_obj.release()
    print('detect down. Wait for post-precessing.')
    return frame_id - 1

