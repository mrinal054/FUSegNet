# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:17:53 2023

@author: mrinal

Save AZH names (Chronic wound dataset) in 10 categories. Categories are created based on the percentage 
groundtruth area in an image. 

% gt-area	No. of images
---------- ---------------
      0	            17
    <0.6	    36
    <0.1	    35
    <0.15	    37
    <0.2	    24
    <0.3	    35
    <0.4	    29
    <0.6	    38
    <10	            20
    ≥10	            07
	
"""
import os
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")

dir_label = r'.\azh_wound_care_center_dataset_patches\test\labels'

names = os.listdir(dir_label)

df = pd.read_excel('percentage.xlsx')

names = df["Name"].tolist()

cat = {}

cat["0"] = names[0:17]
cat["1"] = names[17:53]
cat["2"] = names[53:88]
cat["3"] = names[88:125]
cat["4"] = names[125:149]
cat["5"] = names[149:184]
cat["6"] = names[184:213]
cat["7"] = names[213:251]
cat["8"] = names[251:271]
cat["9"] = names[271:278]

# Save in json
with open("categorized_oldDfu.json", "w") as F:
    json.dump(cat, F)
    
# Read json
with open("categorized_oldDfu.json", "r") as F:
    file = json.load(F)
    
