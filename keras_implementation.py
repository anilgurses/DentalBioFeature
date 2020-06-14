import argparse
from datetime import datetime
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
from tensorflow.keras import layers, models, optimizers


NCLASSES = 2
NUM_CHANNELS = 3
lr = 0.001
ne = 250
HEIGHT = 224
WIDTH = 224

image_feature_description = {
    'img_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'personid': tf.io.FixedLenFeature([], tf.int64),
    'toothid': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_function(proto):
    parsed = tf.io.parse_single_example(proto, image_feature_description)
    img = parsed["img_raw"]
    image = tf.image.decode_png(img, channels=NUM_CHANNELS)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image /= 255.0  # normalize to [0,1] range
    label = parsed["label"]
    toothid = parsed["toothid"]
    personid = parsed["personid"]

    return image, label, toothid, personid


def _train_parser(image, label, toothid, personid):
  return image,label


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Tooth NETWORK TRAINER AND TESTER')
  parser.add_argument("--lr", default=0.0003, type=float, help="Learning Rate")
  parser.add_argument("--ne", default=250,type=int, help="Number of Epochs")
  parser.add_argument("--ft", default="Fine", type=str, help="False Tuning")
  args = parser.parse_args()

  lr = args.lr 
  ne = args.ne 
  ft = args.ft

  dataset = tf.data.TFRecordDataset("train.tfrecord")
  val_dataset = tf.data.TFRecordDataset("val.tfrecord")
  val_dataset = val_dataset.map(_parse_function)
  val_dataset = val_dataset.batch(16)
  dataset = dataset.shuffle(buffer_size=40)
  train_size = int(0.7 * 360)

  train_dataset = dataset.take(train_size)
  test_dataset = dataset.skip(train_size)

  train_dataset = train_dataset.map(_parse_function)
  test_dataset = test_dataset.map(_parse_function)
  train_dataset = train_dataset.batch(32)
  test_dataset = test_dataset.batch(32)

  train_batch = train_dataset.map(_train_parser)
  test_batch = test_dataset.map(_train_parser)


  

  IMG_SHAPE = (224, 224, 3)


  for image_batch, label_batch in train_batch.take(1):
    pass
  

  if ft == "Fine":
    modells = [
      tf.keras.applications.vgg19.VGG19(input_shape=(HEIGHT, WIDTH, NUM_CHANNELS), include_top=False),
      tf.keras.applications.Xception(input_shape=(HEIGHT, WIDTH, NUM_CHANNELS), include_top=False),
      tf.keras.applications.InceptionV3(input_shape=(
          HEIGHT, WIDTH, NUM_CHANNELS), include_top=False),
      tf.keras.applications.MobileNetV2(input_shape=(
          HEIGHT, WIDTH, NUM_CHANNELS), include_top=False)
    ]
  else:
    modells = [
        tf.keras.applications.vgg19.VGG19(input_shape=(
            HEIGHT, WIDTH, NUM_CHANNELS), include_top=False, weights='imagenet'),
        tf.keras.applications.Xception(input_shape=(
            HEIGHT, WIDTH, NUM_CHANNELS), include_top=False, weights='imagenet'),
        tf.keras.applications.InceptionV3(input_shape=(
            HEIGHT, WIDTH, NUM_CHANNELS), include_top=False, weights='imagenet'),
        tf.keras.applications.MobileNetV2(input_shape=(
            HEIGHT, WIDTH, NUM_CHANNELS), include_top=False, weights='imagenet')
    ]
    
  model_names= ["vgg19","xception","inceptionv3","MobileNetv2"]
  last_layers = ["block5_conv4", "conv2d_3", "conv2d_97", "Conv_1"]
  stp = 0
  for base_model in modells:
      
    feature_batch = base_model(image_batch)
    modelName = model_names[stp]

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

  
    checkpoint_path = modelName+".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    if ft == "Fine":
      model.trainable = True
    else:
      model.trainable = False  

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    if ft == "Fine":
      history = model.fit(train_batch,
                        epochs=ne,
                        validation_data=test_batch,
                        callbacks=[cp_callback])
    
    
      hist_df = pd.DataFrame(history.history)
      hist_csv_file = modelName +'_history.csv'
      
      hist_df.to_csv(hist_csv_file)


    layer_output = base_model.get_layer(last_layers[stp]).output
    
    features = []
    ids = []
    tooths = []
    intermediate_model = tf.keras.models.Model(inputs=base_model.input, outputs=layer_output)
    results = {"Id":[],"ToothId":[]}
    for data in val_dataset:
      img, lbl, tid, pid = data
      intermediate_prediction = intermediate_model.predict(img.numpy())

      #pred = base_model(img)

      feature = tf.math.reduce_mean(intermediate_prediction, axis=2)
      feature = tf.math.reduce_mean(feature, axis=1).numpy()
      features.append(feature)

      ids.extend(pid.numpy())
      tooths.extend(tid.numpy())
      


    features = np.concatenate(features)
    features = pd.DataFrame(features)
    features = features.add_prefix('Feature_')
    ids = pd.DataFrame(ids)
    tooths = pd.DataFrame(tooths)

    features['Id'] = ids
    features['ToothId'] = tooths
    if ft == "Fine":
      features.to_csv(modelName+"_"+"feats.csv")
      ids.to_csv(modelName+"_"+"ids.csv")
      tooths.to_csv(modelName+"_"+"tooths.csv")
    else:
      features.to_csv(modelName+"_"+"feats_withoutfine.csv")
      ids.to_csv(modelName+"_"+"ids_withoutfine.csv")
      tooths.to_csv(modelName+"_"+"tooths_withoutfine.csv")
    stp += 1
