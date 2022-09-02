# Dental Biometric Features for Human Identification

This repository includes required software for dental biometric feature extraction. The whole work has done with TF Keras 2.0 and PyTorch 1.5 seperately. Some of the pre-trained models don't exist on each library. Therefore, Keras and PyTorch have used on particular pre-trained networks. 


## Setup

Installing required python packages
> pip3 install -r requirements.txt

All of the setup could be done with above step. Also, you need to install OpenCv for image feeding processes. OpenCv version must be older than version 4.0. The reason for it is the OpenCV has changed some functions(removed some of them) on the newer versions.

## File Organization

The file organization is shown below

```
* Main.py ----> PyTorch implementation for feature extraction. 

* data_load.py ----> PyTorch DataLoader implementation for the training

* create_data.py ----> Creating csv for the data feeding

* keras_implementation.py ----> Keras implementation for feature extraction

* score_analyzer.py ----> Extracting confusion matrix from given result csv

* score_separator.py ----> Purifying the result csv. Dropping some elements such as Tensor([1]) to 1.

* tfrecord_test.py ----> Tensorflow tfrecord file tester

* tfrecordcreator.py ----> Creating tfrecord file for the training.

* Tooth_Dataset.py ----> PyTorch Dataset class implementation for the training

```

## Usage


The main.py can be runned as shown below

```
python3 main.py **args
```

Arguments are listed below

Frozen model path ("models/model4.pth")
* --path  Argument

eval or train 
* --mode 

Learning Rate (0.001)
* --lr 

Number of Epochs (100)
* --ne 

Fine Tuning (Fine)
* --ft 


Example Scenario for Training

```
python3 main.py --mode train --lr 0.0001 --ne 250
```

## Dataset

You can use your own dataset for the training process or pre-trained version of the code. You need to create csv before you train your model. Some path strings should be changed on the code. 

Note: Due to the privacy conservation of biomedical data, the data couldn't be available to public. Please do not request access for the data. 

Contact: anilgurses@ieee.org

