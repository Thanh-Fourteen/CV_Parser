# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:49
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : config.py
import os
import os.path as osp
import datetime


class DBConfig(object):

    STEPS_PER_EPOCH = 1

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 20

    # Backbone network architecture
    # Supported values are: ResNet50
    BACKBONE = "ResNet50"


    # train
    EPOCHS = 30
    INITIAL_EPOCH = 0
    # PRETRAINED_MODEL_PATH = 'weights/db_167_1.9499_1.9947.h5'
    PRETRAINED_MODEL_PATH = "weights/tf_model.h5"
    # PRETRAINED_MODEL_PATH = ''
    LOG_DIR = 'datasets/logs'
    CHECKPOINT_DIR = 'checkpoints'
    LEARNING_RATE = 1e-4


    # dataset
    IGNORE_TEXT = ["*", "###"]

    TRAIN_DATA_PATH = 'datasets/data/train.json'
    VAL_DATA_PATH = 'datasets/data/val.json'

    IMAGE_SIZE = 64
    BATCH_SIZE = 3

    MIN_TEXT_SIZE = 8
    SHRINK_RATIO = 0.4

    THRESH_MIN = 0.3
    THRESH_MAX = 0.7


    def __init__(self):
        """Set values of computed attributes."""

        if not osp.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        self.CHECKPOINT_DIR = osp.join(self.CHECKPOINT_DIR, str(datetime.date.today()))
        if not osp.exists(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)

