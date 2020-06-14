import numpy as np
import tensorflow as tf
from random import shuffle
import glob
import cv2
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys

## Tf 1.4 < 
#tf.enable_eager_execution() 

dataset = tf.data.TFRecordDataset("train.tfrecord")

image_feature_description = {
    'img_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'personid': tf.io.FixedLenFeature([], tf.int64),
    'toothid': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_function(proto):
    parsed = tf.io.parse_single_example(proto, image_feature_description)
    img = parsed["img_raw"]
    label = parsed["label"]
    toothid = parsed["toothid"]
    personid = parsed["personid"]
    return img, label, toothid, personid


def _parse_images(img, label, toothid, personid):
    return img

def _parse_labels(img, label, toothid, personid):
    return label

dataset = dataset.map(_parse_function)

img_dataset = dataset.map(_parse_images)
label_dataset = dataset.map(_parse_labels)



imgs, labels, tooths, p = next(iter(dataset))
print(labels)
img = imgs.numpy()
nparr = np.fromstring(img, np.uint8)
img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cv2.imshow("",img_np)
cv2.waitKey()

# for image_features in dataset:

#     image = image_features['img_raw'].numpy()
#     nparr = np.fromstring(image, np.uint8)
#     img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     #image.set_shape([200, 200, 3])
#     print(img_np)
#     cv2.imshow(" ",img_np)
#     cv2.waitKey()
#     break
