import ast
import os
import sys
import warnings

import pandas as pd
from pandas.api.types import CategoricalDtype

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import json

import numpy as np
from tensorflow.keras.utils import Sequence

from audio_processing import random_crop, random_mask


class DataGenerator(Sequence):
    def __init__(self, path_x_label_list, class_mapping, batch_size=32):
        self.path_x_label_list = path_x_label_list

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.path_x_label_list))
        self.class_mapping = class_mapping
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_x_label_list) / self.batch_size / 10))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_samples = [self.path_x_label_list[k] for k in indexes]

        x, y = self.__data_generation(batch_samples)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        paths, labels = zip(*batch_samples)

        labels = [labels_to_vector(x, self.class_mapping) for x in labels]

        crop_size = np.random.randint(128, 256)

        X = np.array([random_crop(np.load(x), crop_size=crop_size) for x in paths])
        Y = np.array(labels)

        return X, Y[..., np.newaxis]


def get_id_from_path(path):
    base_name = os.path.basename(path)

    return base_name.replace(".wav", "").replace(".npy", "")


def labels_to_vector(labels, mapping):
    vec = [0] * len(mapping)
    vec[mapping[labels]] = 1
    return vec
