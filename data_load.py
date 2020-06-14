import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd
import torchvision
import cv2
from torchvision import transforms, utils, models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.models as models
from os import walk
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from Tooth_Dataset import ToothDataset


def tooth_data_load():

  tooth = pd.read_csv('tooth_dataset.csv')
  same = tooth[tooth["isTooth"] == True].iloc[0:250]
  dif = tooth[tooth["isTooth"] == False]
  
  frames1 = [same, dif]

  train = pd.concat(frames1)
  val_data = pd.read_csv('val_dataset.csv')

  # Transforms for the data loader

  trnsfrm = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(size=(200, 200)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

  tooth_dataset = ToothDataset(dataframe=train, root_dir="Tooth_Data", transform=trnsfrm)
  
  tooth_loader = DataLoader(tooth_dataset, batch_size=4,
                            shuffle=True, num_workers=1,drop_last=True)

  val_dataset = ToothDataset(
      dataframe=val_data, root_dir="Tooth_Data", transform=trnsfrm)
 
  valloader = DataLoader(val_dataset, batch_size=8,  shuffle=False, num_workers=1,drop_last=True)

  return tooth_loader, valloader

def other_tooth_data_load():

    tooth = pd.read_csv('eval_tooth.csv')

    train, val = train_test_split(tooth, test_size=0.2, shuffle=False)
    # Transforms for the data loader

    trnsfrm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    tooth_dataset = ToothDataset(dataframe=train, root_dir="Tooth_Data_TNew",
                    transform=trnsfrm)

    tooth_loader = DataLoader(tooth_dataset, batch_size=64,
                                shuffle=True, num_workers=1, drop_last=True)

    val_dataset = ToothDataset(dataframe=val, root_dir="Tooth_Data_TNew",
                    transform=trnsfrm),
        
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)

    return tooth_loader, valloader
