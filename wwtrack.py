#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @file: wwtrack.py
# @author: jerrzzy
# @date: 2023/7/17


"""
wwtrack.py: WWTrack轮虫追踪软件
"""


import sys
import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qt_material import apply_stylesheet
from database import *
from models import *
from ui import *
import warnings
from multiprocessing import freeze_support


warnings.filterwarnings("ignore")
freeze_support()


class WWTrackMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(WWTrackMainWindow, self).__init__(parent)
        # 数据库
        self.database = Database()
        # 日志记录
        self.logger = get_logger()
        # 设置标题和图标
        self.setWindowTitle('WWTrack 1.0.0 Stable')
        self.setWindowIcon(QIcon("./database/icon/worm.png"))
        # 布局
        self.layout = QGridLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.layout)
        # 设置窗口分辨率
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.5)
        self.width = int(self.screenwidth * 0.5)
        self.resize(self.width, self.height)
        # 文件管理器
        self.file_manager = FileManager(database=self.database, logger=self.logger)
        self.file_manager.file_name_line.currentIndexChanged.connect(self.change_current_file)
        self.layout.addWidget(self.file_manager, 0, 0, 10, 2)
        # 查看器
        self.main_viewer = MainViewer(database=self.database, logger=self.logger)
        self.main_viewer.previous_btn.clicked.connect(self.change_to_previous)
        self.main_viewer.next_btn.clicked.connect(self.change_to_next)
        self.main_viewer.start_btn.clicked.connect(self.pcf_start)
        self.main_viewer.download_btn.clicked.connect(self.download_result)
        self.main_viewer.cuda_btn.clicked.connect(self.set_cuda)
        self.main_viewer.multiple_btn.clicked.connect(self.set_gain_mutilple)
        self.main_viewer.colors_btn.clicked.connect(self.set_colors)
        self.main_viewer.circle_btn.clicked.connect(self.set_circles)
        self.layout.addWidget(self.main_viewer, 0, 2, 10, 7)
        # 线程
        self.pcf_thread = ProcessingCurrentFileThread(database=self.database)
        self.pcf_thread.finished.connect(self.pcf_down)
        self.drf_thread = DownloadResultFileThread(database=self.database)
        # 更新GPU状态
        self.main_viewer.cuda_btn.state = self.database.cfg['cuda']
        self.main_viewer.cuda_btn.update_icon()

    def change_current_file(self, _id):
        """切换当前数据"""
        self.database.current_obj_id = _id
        self.main_viewer.change_current_file()
        file_obj = self.database.current_file_obj()
        self.logger.info(f'User: Change Current File: {file_obj["name"]}')

    def change_to_previous(self):
        """切换到上一个"""
        if self.database.get_length() > 0 and 0 < self.database.current_obj_id < self.database.get_length():
            self.file_manager.file_name_line.setCurrentIndex(self.database.current_obj_id - 1)

    def change_to_next(self):
        """切换到下一个"""
        if self.database.get_length() > 0 and 0 <= self.database.current_obj_id < self.database.get_length()-1:
            self.file_manager.file_name_line.setCurrentIndex(self.database.current_obj_id + 1)

    def pcf_start(self):
        """PCF线程开始执行"""
        if self.database.get_length() > 0:
            self.pcf_thread.start()
            self.file_manager.set_progressbar(state=True)
            self.logger.info(f'User: Start Processing Thread')
        else:
            QMessageBox.warning(self, 'WARNING', 'Please input file!', QMessageBox.Close)

    def pcf_down(self):
        """PCF线程结束"""
        self.file_manager.set_progressbar(state=False)
        self.change_current_file(self.database.current_obj_id)
        self.logger.info('System: Processing Thread Down')

    def download_result(self):
        """下载结果"""
        if self.database.get_length() > 0:
            if not self.database.running:
                now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
                fileName_choose, filetype = QFileDialog.getSaveFileName(self, "Save Result to zip", f'C:/wwtrack_{now_time}.zip', "ZIP Files (*.zip)")
                self.database.zip_file_path = fileName_choose
                if fileName_choose == "":
                    return
                else:
                    self.drf_thread.start()
                    self.logger.info(f'User: Save Result to {fileName_choose}')
            else:
                QMessageBox.warning(self, 'WARNING', 'Please wait for processing down!', QMessageBox.Close)
        else:
            QMessageBox.warning(self, 'WARNING', 'Unable to Download Result!', QMessageBox.Close)

    def set_cuda(self):
        self.main_viewer.cuda_btn.click()
        if self.main_viewer.cuda_btn.state:
            self.database.cfg['cuda'] = True
            self.logger.info(f'User: Activate GPU')
        else:
            self.database.cfg['cuda'] = False
            self.logger.info(f'User: Shut Down GPU')
        self.database.save_config()

    def set_gain_mutilple(self):
        value, ok = QInputDialog.getDouble(self, "GainMultipleInput", "请输入速度换算系数:", self.database.cfg['gain_multiple'],
                                           -9999999, 9999999, 2)
        if ok:
            self.database.cfg['gain_multiple'] = value
            self.database.save_config()
            self.logger.info(f'User: Save config down.')

    def set_colors(self):
        c = QColorDialog.getColor()
        colors = []
        for i in range(QColorDialog.customCount()):
            qcolor = QColorDialog.customColor(i)
            opencv_bgr = (qcolor.blue(), qcolor.green(), qcolor.red())
            colors.append(opencv_bgr)
        self.database.cfg['colors'] = colors
        self.database.save_config()
        self.logger.info(f'User: Save config down.')

    def set_circles(self):
        value_list = [self.database.cfg['confidence']] + self.database.cfg['center'] + self.database.cfg['radius'] + self.database.cfg['export_circles']
        set_dialog = MySetCirclesDialog(self, value_list)
        set_dialog.show()
        set_dialog.ok_button.clicked.connect(set_dialog.accept)
        if set_dialog.exec() == QDialog.Accepted:
            value_list = set_dialog.get_value()
            self.database.cfg['confidence'] = value_list[0]
            self.database.cfg['center'] = [value_list[1], value_list[2]]
            self.database.cfg['radius'] = [value_list[3], value_list[4], value_list[5]]
            self.database.cfg['export_circles'] = [value_list[6], value_list[7], value_list[8]]
            self.database.save_config()
            if self.database.get_length() > 0:
                self.change_current_file(self.database.current_obj_id)
            self.logger.info(f'User: Save config down.')


class MySetCirclesDialog(QDialog):
    def __init__(self, parent, value_list):
        super(MySetCirclesDialog, self).__init__(parent)
        # 数值列表
        self.value_list = value_list
        # 设置窗口标题和位置
        self.setWindowTitle('SetCirclesInput')
        # self.desktop = QApplication.desktop()
        # self.screenRect = self.desktop.screenGeometry()
        # self.screenheight = self.screenRect.height()
        # self.screenwidth = self.screenRect.width()
        # self.height = int(self.screenheight * 0.3)
        # self.width = self.height
        # self.resize(self.width, self.height)
        # 创建表单布局
        self.form_layout = QFormLayout()
        doubleValidator = QDoubleValidator(self)
        doubleValidator.setRange(0, 1)
        doubleValidator.setNotation(QDoubleValidator.StandardNotation)
        doubleValidator.setDecimals(2)
        self.value_dict = ['confidence', 'center_x', 'center_y', 'radius_1', 'radius_2', 'radius_3']
        self.input_widget_list = []
        for i in range(len(self.value_dict)):
            # input_line = QLineEdit()
            # # input_line.setPlaceholderText(value_dict[i])
            # input_line.setValidator(doubleValidator)
            input_line = QDoubleSpinBox()
            if i < 3:
                # 前两个可以设置为
                input_line.setRange(0, 1)
            else:
                # 其它的最大0.5
                input_line.setRange(0, 0.5)
            input_line.setSingleStep(0.01)
            input_line.setValue(self.value_list[i])
            self.form_layout.addRow(self.value_dict[i], input_line)
            self.input_widget_list.append(input_line)
        # button = QPushButton('ok')
        self.r1_box = QCheckBox()
        self.r1_box.setChecked(value_list[5])
        self.r2_box = QCheckBox()
        self.r2_box.setChecked(value_list[6])
        self.r3_box = QCheckBox()
        self.r3_box.setChecked(value_list[7])
        self.form_layout.addRow('export r1', self.r1_box)
        self.form_layout.addRow('export r2', self.r2_box)
        self.form_layout.addRow('export r3', self.r3_box)
        self.ok_button = QDialogButtonBox(QDialogButtonBox.Ok)
        self.form_layout.addRow(self.ok_button)
        self.setLayout(self.form_layout)

    def get_value(self):
        float_value = [round(input_line.value(), 2) for input_line in self.input_widget_list]
        state_value = [bool(self.r1_box.checkState()), bool(self.r2_box.checkState()), bool(self.r3_box.checkState())]
        return float_value + state_value


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    win = WWTrackMainWindow()
    win.show()
    sys.exit(app.exec_())