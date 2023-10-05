import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

data = DataLoader.from_folder('rice/')
train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)

accuracy = model.evaluate(test_data)


model.export(export_dir='.', tflite_filename='rice.tflite',
             label_filename='labels.txt',)
