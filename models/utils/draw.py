#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @file: draw.py
# @author: jerrzzy
# @date: 2023/7/13


import os
import numpy as np
import cv2


def mot_visualize_video(src_video_path, out_video_path, trk_seq_path, colors, center_list, radius_list, export_list, fps=30, w=1280, h=960):
    """
    :param src_video_path: 原始视频路径
    :param out_video_path: 输出视频路径
    :param trk_seq_path: MOT追踪序列文件
    :param colors: 绘制轨迹的颜色列表
    :param center_list: 中心点偏移比例值
    :param radius_list: 三个标记位置圆圈的比例值
    :param export_list: 三个标记位置圆圈是否导出
    :param fps: 输出视频的FPS
    :param w: 输出视频的宽
    :param h: 输出视频的长
    :return: None
    """
    trk_data = np.loadtxt(trk_seq_path, dtype=int, delimiter=',')
    if not len(trk_data) > 0:
        return None
    frame_id = 1
    src = cv2.VideoCapture(src_video_path)
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    bg = np.zeros((h, w, 3), np.uint8)
    trajectory_num = int(trk_data[:, 1].max())
    trajectory_temp = np.zeros((trajectory_num, 2))  # 点的缓存值，用来绘制曲线
    is_trajectory_find = [False] * trajectory_num  # 是否已经找到了第一个点
    while src.isOpened():
        ret, frame = src.read()
        if not ret:
            break
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        h, w, _ = frame.shape
        for i in range(3):
            if export_list[i]:
                cv2.circle(frame, (int(w*center_list[0]), int(h*center_list[1])), int(radius_list[i]*min(h, w)), color_list[i], 2)
        cv2.putText(frame, str(frame_id), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 3)
        frame_data = trk_data[trk_data[:, 0] == frame_id]
        for _, worm_id, x, y, w, h, _, _, _, _ in frame_data:
            cv2.rectangle(frame, (y, x), (y + h, x + w), colors[(worm_id-1) % 3], 2)  # 方框
            cv2.putText(frame, str(worm_id), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # ID号
            cx, cy = int(x+w/2), int(y+h/2)
            if is_trajectory_find[worm_id-1]:
                (cx0, cy0) = trajectory_temp[worm_id-1]
                cv2.line(bg, (int(cy0), int(cx0)), (int(cy), int(cx)), colors[(worm_id-1) % len(colors)], 2)
            else:
                is_trajectory_find[worm_id - 1] = True
            trajectory_temp[worm_id-1] = np.array([cx, cy])
        frame = cv2.addWeighted(frame, 0.8, bg, 0.2, 0)
        # cv2.imshow('demo', frame)
        # cv2.waitKey(1)
        out.write(frame)
        frame_id += 1
    src.release()
    out.release()
    cv2.destroyAllWindows()


def mot_visualize_image(out_image_dir, trk_seq_path, colors, center_list, radius_list, export_list, w=1280, h=960):
    """
    :param out_image_dir: 输出图片目录
    :param trk_seq_path: MOT追踪序列文件
    :param colors: 绘制轨迹的颜色列表
    :param center_list: 中心点偏移比例值
    :param radius_list: 三个标记位置圆圈的比例值
    :param export_list: 三个标记位置圆圈是否导出
    :param w: 输出视频的宽
    :param h: 输出视频的长
    :return: None
    """
    trk_data = np.loadtxt(trk_seq_path, dtype=int, delimiter=',')
    if not len(trk_data) > 0:
        return None
    trajectory_num = int(trk_data[:, 1].max())
    if trajectory_num > 0:
        trajectory_image_list = [255 * np.ones((h, w, 3), np.uint8) for t in range(trajectory_num+1)]
        total_frames_list = np.zeros(trajectory_num+1)
        for worm_id in range(1, trajectory_num+1):
            total_frames_list[worm_id] = trk_data[trk_data[:, 1] == worm_id][:, 0].max()
            id_data = trk_data[trk_data[:, 1] == worm_id]
            _, _, x0, y0, w0, h0, _, _, _, _ = id_data[0]
            cx0, cy0 = int(x0+w0/2), int(y0+h0/2)
            for _, _, x, y, w, h, _, _, _, _ in id_data:
                cx, cy = int(x+w/2), int(y+h/2)
                cv2.line(trajectory_image_list[0], (cy0, cx0), (cy, cx), colors[(worm_id - 1) % 3], 2)
                cv2.line(trajectory_image_list[worm_id], (cy0, cx0), (cy, cx), colors[(worm_id - 1) % len(colors)], 2)
                cx0 = cx
                cy0 = cy
        for i, trajectory_image in enumerate(trajectory_image_list):
            # 绘制圆心
            """"""
            color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            h, w, _ = trajectory_image.shape
            for j in range(3):
                if export_list[j]:
                    cv2.circle(trajectory_image, (int(w * center_list[0]), int(h * center_list[1])), int(radius_list[j] * min(h, w)), color_list[j], 2)
            """"""
            cv2.imwrite(os.path.join(out_image_dir, f'cache{i}.png'), trajectory_image)