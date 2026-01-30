#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @CreateTime : 2026/01/30 16:21
# @Author     : wephiles@wephiles
# @IDE        : PyCharm
# @ProjectName: ImageRecognition
# @FileName   : ImageRecognition/data_preparation.py
# @Description: This is description of this script.
# @Interpreter: python 3.0+
# @Motto      : You must take your place in the circle of life!
# @AuthorSite : https://github.com/wephiles or https://gitee.com/wephiles

import os
import shutil
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2


class DataPreparer(object):
    def __init__(
            self,
            raw_data_path: str = os.path.join('data', 'raw'),  # data/raw
            output_path: str = os.path.join('data', 'splits')  # data/splits
    ):
        self.raw_data_path = raw_data_path
        self.output_path = output_path

        self.classes = ['wheat', 'weed']  # [小麦, 杂草]

        # 创建输出目录
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                os.makedirs(os.path.join(output_path, split, class_name), exist_ok=True)

    def prepare_data(self, test_size=0.15, val_size=0.15, seed=42):
        """准备数据并划分训练集、验证集和测试集。

        Args:
            test_size ():
            val_size ():
            seed ():

        Returns:

        """
        # 收集所有文件路径和标签
        image_paths = []
        labels = []

        # 小麦图片
        wheat_dir = os.path.join(self.raw_data_path, 'wheat')
        for image_name in os.listdir(wheat_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(wheat_dir, image_name))
                labels.append(0)  # 0 表示小麦

        # 杂草图片(合并看麦娘和野燕麦)
        weed_dirs = ['weed_alopecurus', 'weed_avena']
        for weed_dir in weed_dirs:
            weed_path = os.path.join(self.raw_data_path, weed_dir)
            if os.path.exists(weed_path):
                for image_name in os.listdir(weed_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(weed_path, image_name))
                        labels.append(1)  # 0 代表野草
        print('图片准备工作完毕!')
        print('总图片数:', len(image_paths))
        print('小麦图片数:', labels.count(0))
        print('杂草图片数:', labels.count(1))

        # 划分训练集、验证集和测试集
        # 第一次划分: 分出测试集
        X_temp, X_text, y_temp, y_test = train_test_split(

        )
    pass

# --END--
