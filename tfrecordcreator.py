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




tooth = pd.read_csv('tooth_dataset.csv')
same = tooth[tooth["isTooth"] == True].iloc[0:250]
dif = tooth[tooth["isTooth"] == False]

frames1 = [same, dif]

train = pd.concat(frames1)
val_data = pd.read_csv('val_dataset.csv')


print("Creating the training tfrecord file")

recordFileName = ("train.tfrecord")
# tfrecord file writer
writer = tf.io.TFRecordWriter(recordFileName)

for index, val in train.iterrows():
    img = os.path.join("Tooth_Data", val["img"])
    img_path = os.path.join(img)
    img_raw = open(img_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(img_raw).shape

    label = 0
    if val["isTooth"]:
        label = 1
    personid = int(val["personid"])
    toothid = int(val["toothid"])

    example = tf.train.Example(features=tf.train.Features(
        feature={
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "personid": tf.train.Feature(int64_list=tf.train.Int64List(value=[personid])),
            "toothid": tf.train.Feature(int64_list=tf.train.Int64List(value=[toothid]))
        }))
    writer.write(example.SerializeToString())
writer.close()

sys.stdout.flush()

print("Creating the validation tfrecord file")

recordFileName = ("val.tfrecord")
# tfrecord file writer
writer = tf.io.TFRecordWriter(recordFileName)

for index, val in val_data.iterrows():
    img = os.path.join("Tooth_Data", val["img"])
    img_path = os.path.join(img)
    img_raw = open(img_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(img_raw).shape

    label = 0
    if val["isTooth"]:
        label = 1
    personid = int(val["personid"])
    toothid = int(val["toothid"])

    example = tf.train.Example(features=tf.train.Features(
        feature={
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "personid": tf.train.Feature(int64_list=tf.train.Int64List(value=[personid])),
            "toothid": tf.train.Feature(int64_list=tf.train.Int64List(value=[toothid]))
        }))
    writer.write(example.SerializeToString())

writer.close()

