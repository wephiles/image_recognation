#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @CreateTime : 2026/01/31 17:17
# @Author     : wephiles@wephiles
# @IDE        : PyCharm
# @ProjectName: ImageRecognition
# @FileName   : ImageRecognition/dataset.py
# @Description: This is description of this script.
# @Interpreter: python 3.0+
# @Motto      : You must take your place in the circle of life!
# @AuthorSite : https://github.com/wephiles or https://gitee.com/wephiles

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class WheatWeedDataset(Dataset):

    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # 加载小麦照片
        pass

        # 加载杂草照片
        pass

# --END--
