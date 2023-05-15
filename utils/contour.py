# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:21:11 2023

@author: mrinal
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 

GT_AVAILABLE = True # set true if ground truth is available

# AZH Chronic wound dataset directory
dir_im = r'.\azh_wound_care_center_dataset_patches\test\images' # image directory
dir_label = r'.\azh_wound_care_center_dataset_patches\test\labels' # label directory
dir_pred = r'.\predictions\OldDFU' # prediction directory

name = 'e7d099b05fc99c39b28a3557acc12837_0.png' # image name

# Read images
im = cv2.imread(os.path.join(dir_im, name))[:,:,::-1]
pred_mask = cv2.imread(os.path.join(dir_pred, name), 0)
  
contour_im = im.copy()
 
# Find contours
if GT_AVAILABLE:
    gt_mask = cv2.imread(os.path.join(dir_label, name), 0)
    contours_gt, hierarchy_gt = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found in GT = " + str(len(contours_gt)))
    cv2.drawContours(contour_im, contours_gt, -1, (255, 0, 0), 1) # -1 signifies drawing all contours

contours_pred, hierarchy_pred = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found in prediction = " + str(len(contours_pred)))
cv2.drawContours(contour_im, contours_pred, -1, (0, 255, 0), 1) # -1 signifies drawing all contours

plt.title('Contours of the prediction')
plt.xticks([])
plt.yticks([])

plt.imshow(contour_im)

# Save image
cv2.imwrite(os.path.join(dir_pred, 'contour_' + name), contour_im[:,:,::-1])
