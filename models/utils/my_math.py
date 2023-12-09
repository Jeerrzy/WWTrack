#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @file: my_math
# @author: jerrzzy
# @date: 2023/7/13


import os
import json
import math
import numpy as np
import xlsxwriter as xw
from matplotlib import pyplot as plt


def get_math_results(result_dir, trk_seq_path, center_list, radius_list, w=1280, h=960):
    """
    :param result_dir: 保存图像输出图像目录
    :param trk_seq_path: MOT格式追踪序列数据
    :param center_list: 中心点偏移比例值
    :param radius_list: 三个标记位置圆圈的比例值
    :param w: 输出视频的宽
    :param h: 输出视频的长
    :return: None
    """
    result_dict = dict()
    trk_data = np.loadtxt(trk_seq_path, dtype=int, delimiter=',')
    if not len(trk_data) > 0:
        return None
    trajectory_num = int(trk_data[:, 1].max())
    if trajectory_num > 0:
        user_cx, user_cy = center_list[0] * h, center_list[1] * w
        r1, r2, r3 = radius_list[0] * min(h, w), radius_list[1] * min(h, w), radius_list[2] * min(h, w)
        for worm_id in range(1, trajectory_num + 1):
            id_dict = dict()
            frames, centers, positions, displacements, angulars = [], [], [], [], []
            id_data = trk_data[trk_data[:, 1] == worm_id]
            _, _, x0, y0, w0, h0, _, _, _, _ = id_data[0]
            cx0, cy0 = int(x0+w0/2), int(y0+h0/2)
            for f, _, x, y, w, h, _, _, _, _ in id_data:
                cx, cy = int(x + w / 2), int(y + h / 2)
                frames.append(f)
                centers.append([cx, cy])
                # 判断位置
                d = math.sqrt((cx-user_cx)**2+(cy-user_cy)**2)
                if d <= r1:
                    positions.append('r1')
                elif r1 < d <= r2:
                    positions.append('r2')
                elif r2 < d <= r3:
                    positions.append('r3')
                else:
                    positions.append('N/A')
                # 计算直线位移
                displacements.append(math.sqrt((cx-cx0)**2+(cy-cy0)**2))
                # 计算旋转角度
                if int(cx) == int(cx0) and int(cy) == int(cy0):
                    angular = 0.0
                else:
                    x0_convert, y0_convert = (cy0 - user_cy), (user_cx - cx0)
                    x_convert, y_convert = (cy - user_cy), (user_cx - cx)
                    try:
                        angular = np.rad2deg(np.arccos((x0_convert * x_convert + y0_convert * y_convert) / (
                                    math.sqrt(x0_convert ** 2 + y0_convert ** 2) * math.sqrt(
                                x_convert ** 2 + y_convert ** 2))))
                    except:
                        try:
                            angular = angulars[-1]
                        except:
                            angular = 0.0
                angulars.append(angular)
                cx0, cy0 = cx, cy
            id_dict['frame'] = np.array(frames)
            id_dict['center'] = np.array(centers)
            id_dict['position'] = positions
            id_dict['displacement'] = np.array(displacements)
            # 计算速度
            speeds = np.zeros(len(frames))
            speeds[0] = displacements[1]
            speeds[-1] = displacements[-1]
            for i in range(1, len(frames)-1):
                speeds[i] = (displacements[i] + displacements[i+1])/2
            id_dict['speed'] = speeds
            # 计算加速度
            speeds = list(speeds)
            speeds.insert(0, speeds[0])
            speeds_diff = np.diff(np.array(speeds))
            accelerations = np.zeros(len(frames))
            accelerations[0] = speeds_diff[1]
            accelerations[-1] = speeds_diff[-1]
            for i in range(1, len(frames)-1):
                accelerations[i] = (speeds_diff[i] + speeds_diff[i+1])/2
            id_dict['acceleration'] = accelerations
            # 角度
            id_dict['angular'] = np.array(angulars)
            # 角速度
            angular_speeds = np.zeros(len(frames))
            angular_speeds[0] = angulars[1]
            angular_speeds[-1] = angulars[-1]
            for i in range(1, len(frames)-1):
                angular_speeds[i] = (angulars[i] + angulars[i+1])/2
            id_dict['angular_speed'] = angular_speeds
            # 角加速度
            angular_speeds = list(angular_speeds)
            angular_speeds.insert(0, angular_speeds[0])
            angular_speeds_diff = np.diff(np.array(angular_speeds))
            angular_accelerations = np.zeros(len(frames))
            angular_accelerations[0] = angular_speeds_diff[1]
            angular_accelerations[-1] = angular_speeds_diff[-1]
            for i in range(1, len(frames)-1):
                angular_accelerations[i] = (angular_speeds_diff[i] + angular_speeds_diff[i+1])/2
            id_dict['angular_acceleration'] = angular_accelerations
            # 保存图片
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax1.set_xlabel('frames')
            ax1.set_ylabel('speed')
            ax2 = fig.add_subplot(222)
            ax2.set_xlabel('frames')
            ax2.set_ylabel('acceleration')
            ax3 = fig.add_subplot(223)
            ax3.set_xlabel('frames')
            ax3.set_ylabel('angular speed')
            ax4 = fig.add_subplot(224)
            ax4.set_xlabel('frames')
            ax4.set_ylabel('angular acceleration')
            ax1.plot(id_dict['frame'], id_dict['speed'])
            ax2.plot(id_dict['frame'], id_dict['acceleration'])
            ax3.plot(id_dict['frame'], id_dict['angular_speed'])
            ax4.plot(id_dict['frame'], id_dict['angular_acceleration'])
            plt.savefig(os.path.join(result_dir, f'v_a-frame_cache{worm_id}.png'))
            # 写excel
            workbook = xw.Workbook(os.path.join(result_dir, f'math_cache{worm_id}.xlsx'))
            worksheet = workbook.add_worksheet('worm'+str(worm_id))
            worksheet.activate()
            header_names = ['Frames', 'Center', 'Position', 'Speed', 'Acceleration', 'Angular', 'Angular_Acceleration']
            worksheet.set_column('A:B', 10)
            worksheet.set_column('B:C', 25)
            worksheet.set_column('C:D', 10)
            worksheet.set_column('D:E', 15)
            worksheet.set_column('E:F', 15)
            worksheet.set_column('F:G', 15)
            worksheet.set_column('G:H', 15)
            head_format = {
                'font_size': 15,  # 字体大小
                'bold': True,  # 是否粗体
                'font_color': '#9400D3',  # 字体颜色
                'align': 'center',  # 水平居中对齐
                'valign': 'vcenter',  # 垂直居中对齐
                'border': 1,  # 边框宽度
                'top': 1,  # 上边框
                'left': 1,  # 左边框
                'right': 1,  # 右边框
                'bottom': 1  # 底边框
            }
            content_format = {
                'font_size': 10,  # 字体大小
                'align': 'center',  # 水平居中对齐
                'valign': 'vcenter'  # 垂直居中对齐
            }
            head_style = workbook.add_format(head_format)
            worksheet.write_row('A1', header_names, head_style)
            content_style = workbook.add_format(content_format)
            for i in range(len(frames)):
                worksheet.write(i + 1, 0, str(id_dict['frame'][i]), content_style)
                worksheet.write(i + 1, 1, str(id_dict['center'][i]), content_style)
                worksheet.write(i + 1, 2, str(id_dict['position'][i]), content_style)
                worksheet.write(i + 1, 3, str(id_dict['speed'][i]), content_style)
                worksheet.write(i + 1, 4, str(id_dict['acceleration'][i]), content_style)
                worksheet.write(i + 1, 5, str(id_dict['angular_speed'][i]), content_style)
                worksheet.write(i + 1, 6, str(id_dict['angular_acceleration'][i]), content_style)
            workbook.close()
            result_dict[str(worm_id)] = id_dict
        with open(os.path.join(result_dir, 'result.json'), 'w') as f:
            json.dump(result_dict, f, indent=2, cls=NumpyArrayEncoder)


class NumpyArrayEncoder(json.JSONEncoder):
    """重写JSONEncoder中的default方法"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
