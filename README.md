# FUSegNet

## Description will be added soon

FUSegNet and x-FUSegNet are implemented on top of [qubvel's](https://github.com/qubvel/segmentation_models.pytorch) implementation.  

## Saved models
* [FUSegNet](https://drive.google.com/drive/folders/14HFRiNdeN10NPx7S6Lts4ymidNpjibI2?usp=sharing) trained on Chronic Wound dataset
* [xFUSegNet](https://drive.google.com/drive/folders/18696pUMWWdIOAgOLcXR_hut0ukKPXuV9?usp=sharing) trained on MICCAI FUSeg Challenge 2021 dataset

## Code description

* utils <br>
	|--`category.py`: Lists AZH test dataset names into 10 categories. Categories are created based on %GT area in images.<br>
	|--`eval.py`: Performs data-based evaluation.<br>
	|--`eval_categorically.py`: Performs data-based evaluation for each category.<br>
	|--`eval_boxplot.py`: Performs image-based evaluation for each category that is required for boxplot. The final output is 
	an excel file with multiple sheets. Each sheet stores results for a perticular category.<br>
	|--`boxplot.py`: Creates a boxplot. It utilizes the excel file generated by `eval_boxplot.py`.<br>
	|--`contour.py`: Draws contours around the wound region.<br>
	|--`runtime_patch.py`: Creates patch during runtime. <br>
* `fusegnet_all.py`: It's an end-to-end file contains codes for dataloader, training and testing using the FUSegNet model.
* `fusegnet_train.py`: It is to train a dataset using the FUSegNet model.
* `fusegnet_test.py`: It is to perform inference using the FUSegNet model.
* `xfusegnet_all.py`: It's an end-to-end file contains codes for dataloader, training and testing using the xFUSegNet model.
* `xfusegnet_train.py`: It is to train a dataset using the xFUSegNet model.
* `xfusegnet_test.py`: It is to perform inference using the xFUSegNet model.
* `FUSegNet_feature_visualization.ipynb`: Demonstrates intermediate features.

## Directory setup

## Parameters setup

## How to use

## Results

## Reference
