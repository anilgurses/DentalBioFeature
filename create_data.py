# -*- coding: utf-8 -*-
import pandas as pd
import os
from os import walk
from tqdm import tqdm
import random


def create_csv():
    
    data = {"img": [], "personid": [], "isTooth": [], "toothid": []}

    persons = pd.read_excel("kisiler.xlsx", names=[
        "name", "temp", "imgid", "personid", "temp", "temp"],header=None)

    val = pd.read_csv("dene.csv", names=[
                      "id"], header=None).iloc[:, 0].to_list()

    persons = persons.drop(["name", "temp", "temp.1", "temp.2"], axis=1)

    ## Acquiring Image Data
    mainImgs = []
    for (dirpath, dirnames, filenames) in walk("Tooth_Data/"):
        mainImgs.extend(filenames)
        break
    
    val_data = {"img": [], "personid": [], "isTooth": [], "toothid":[]}

    for img in tqdm(mainImgs):

        img_id = img.split("-")[0]
        tooth_id = img.split("-")[1].split(".")[0]

        if int(img_id) in val:
            
            if int(tooth_id) < 11:
                data["img"].append(img)
                data["isTooth"].append(False)
                data["personid"].append("0")
                data["toothid"].append(tooth_id)
            else:
                val_data["img"].append(img)
                val_data["isTooth"].append(True)
                prsn_id = persons[persons["imgid"] ==int(img_id)]["personid"].iloc[0]
                val_data["personid"].append(str(prsn_id))
                val_data["toothid"].append(tooth_id)
        else:
            data["img"].append(img)
            if int(tooth_id) < 11:
                data["isTooth"].append(False)
                data["personid"].append("0")
                data["toothid"].append(tooth_id)
            else:
                data["isTooth"].append(True)
                prsn_id = persons[persons["imgid"] ==
                                  int(img_id)]["personid"].iloc[0]
                data["personid"].append(str(prsn_id))
                data["toothid"].append(tooth_id)
         
    cmp_df = pd.DataFrame(
        data, columns=['img', 'personid', 'toothid','isTooth'])
    cmp_df.to_csv('tooth_dataset.csv')

    val_df = pd.DataFrame(
        val_data, columns=['img', 'personid', 'toothid', 'isTooth'])
    val_df.to_csv('val_dataset.csv')
    
create_csv()
