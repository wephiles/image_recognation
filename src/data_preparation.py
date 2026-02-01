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
import sys

from matplotlib.sphinxext.figmpl_directive import figurempl_addnode
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2


class DataPreparer(object):
    def __init__(
            self,
            raw_data_path: str = os.path.join('data', 'raw'),  # data/raw
            output_path: str = os.path.join('data', 'splits')  # data/splits
    ):
        self.raw_data_path = raw_data_path  # ./data/raw
        self.output_path = output_path  # ./data/splits

        self.classes = ['wheat', 'weed']  # [小麦, 杂草]

        # 创建输出目录
        # 此语句块的作用是创建出下面这样的目录结构:
        # splits/
        #   train/
        #       weed/
        #       wheat/
        #   val/
        #       weed/
        #       wheat/
        #   test/
        #       weed/
        #       wheat/
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:  # self.classes = ['wheat', 'weed']
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
        wheat_dir = os.path.join(self.raw_data_path, 'wheat')  # ./data/raw/wheat
        for image_name in os.listdir(wheat_dir):  # image_name = 'wheat_id_1.png'
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # image_paths = ['./data/raw/wheat/wheat_id_1.png', './data/raw/wheat/wheat_id_2.png', ...]
                # labels = [0, 0, ...]
                image_paths.append(os.path.join(wheat_dir, image_name))
                labels.append(0)  # 0 表示小麦
        # image_paths = ['./data/raw/wheat/wheat_id_1.png', './data/raw/wheat/wheat_id_2.png', ...]
        # labels = [0, 0, ...]

        # 杂草图片(合并看麦娘和野燕麦)
        weed_dirs = ['weed_alopecurus', 'weed_avena']
        for weed_dir in weed_dirs:
            weed_path = os.path.join(self.raw_data_path, weed_dir)  # ./data/raw/weed_alopecurus 和 ./data/raw/weed_avena
            if os.path.exists(weed_path):
                for image_name in os.listdir(weed_path):  # image_name = 'weed_id_1.png'
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(weed_path, image_name))
                        labels.append(1)  # 1 代表野草
        # image_paths = ['./data/raw/weed/weed_id_1.png', './data/raw/weed/weed_id_2.png', ...]
        # labels = [0, 0, ...]

        print('图片准备工作完毕!')
        print('总图片数:', len(image_paths))
        print('小麦图片数:', labels.count(0))
        print('杂草图片数:', labels.count(1))

        # 划分训练集、验证集和测试集
        # 第一次划分: 分出测试集
        # X 为特征 y为标签
        """关于 scikit-learn 的 train_test_split 函数:
        
        train_test_split 是 scikit-learn 中 model_selection 模块的一个核心函数，用于将数据集随机划分为训练集和测试集（有时还包括
        验证集）。这是机器学习工作流程中至关重要的一步，用于评估模型在未见数据上的性能。
        
        主要目的：防止模型过拟合，通过将数据分为训练集（用于训练模型）和测试集（用于评估模型性能）来客观评估模型的泛化能力。
        
        函数签名:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=None, 
            train_size=None, 
            random_state=None, 
            shuffle=True, 
            stratify=None
        )
        
        参数解释:
        下面两个参数是必须要有的:
        X: 特征数据，通常是一个二维数组或者pandas.DataFrame
        y: 标签
        
        test_size: float/int， 默认为 None, 常用的值有 0.2， 0.25， 0.3
            float: 表示测试集占整个数据集的比例 [0.0, 1.0]
            int: 表示测试集的绝对样本数
            None: 使用 train_size 的补集, 即 1 - train_size
        train_size: float/int， 默认为 None
            float: 表示训练集占整个数据集的比例 [0.0, 1.0]
            int: 表示训练集的绝对样本数
            如果同时使用 test_size 和 train_size，要确保 train_size + test_size <= 1.0
        
        random_state: int/RandomState 实例, 默认为 None
            控制随机数生成器的种子
            设置固定值可以确保结果可重复, 在需要可重复实验中非常重要
        shuffle: bool, 默认为 True
            控制是否在分割前打乱数据
            对于时间序列数据，通常设置为 False
            如果设置为 False, 那么 stratify 必须为 None
        stratify: array-like, 默认为 None
            确保训练集和测试集中各类比例与原始数据集相同
            通常传入目标变量 y
            对于不平衡数据集特别有用
        
        返回值:
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集标签  
            y_test: 测试集标签
            如果传入多个数组，会按照相同的顺序返回划分后的数组。
            
        示例:
        1. 基本划分
            ```python
            import numpy as np
            from sklearn.model_selection import train_test_split
            
            # 创建示例数据
            X = np.arange(100).reshape((50, 2))  # 创建了一个 50 行 2 列的矩阵(二维数组)用作特征
            y = np.arange(50)  # 创建了一个长度为 50 的一维数组用作标签
            
            # 基本划分: 75% 训练 25% 测试
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25,
                random_state=42  # 可以设置种子，这样的话结果可以复现
            )
            
            print(f"原始数据形状: X={X.shape}, y={y.shape}")
            print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
            ```
        2. 分层抽样 -- 用于分类问题
            ```python
            
            ```
        """
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels,
            test_size=test_size, stratify=labels, random_state=seed
        )

        # 第二次划分: 从剩余数据集中分出验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio, stratify=y_temp, random_state=seed
        )

        print('训练集长度:', len(X_train))
        print('验证集长度:', len(X_val))
        print('测试集长度:', len(X_test))

        self._copy_files(X_train, y_train, 'train')
        self._copy_files(X_val, y_val, 'val')
        self._copy_files(X_test, y_test, 'test')

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _copy_files(self, file_paths, labels, split_name):
        """复制文件到目标目录.

        Args:
            file_paths ():
            labels ():
            split_name ():

        Returns:

        """
        for file_path, label in zip(file_paths, labels):
            class_name = 'wheat' if label == 0 else 'weed'
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(self.output_path, split_name, class_name, file_name)

            try:
                shutil.move(file_path, dest_path)
            except Exception as e:
                print('复制文件失败, 错误信息:', e, file=sys.stderr)
        print(split_name, '集复制完成.')

    @staticmethod
    def visualize_distribution(labels):
        import matplotlib.pyplot as plt

        # 让 matplotlib 使用中文黑体（SimHei）
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号 "-" 显示为方块的问题

        wheat_count = labels.count(0)
        weed_count = labels.count(1)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(['小麦', '杂草'], [wheat_count, weed_count], color=['gold', 'green'])
        plt.title('数据分布')
        plt.ylabel('数量')

        # 在柱状图上显示数量
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}', ha='center', va='bottom',
            )
        plt.tight_layout()
        plt.savefig('distribution.png')
        plt.show()


if __name__ == '__main__':
    preparer = DataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.prepare_data()
    preparer.visualize_distribution(y_train + y_val + y_test)

# --END--
