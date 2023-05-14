import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses, base
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import scipy.io as sio

import warnings
warnings.filterwarnings("ignore")

"""## Dataloader"""

class Dataset(BaseDataset):
    """Reference: https://github.com/qubvel/segmentation_models.pytorch
    
    Args:
        list_IDs (list): List of image names with extension
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            list_IDs,
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
              
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)    # ----------------- pay attention ------------------ #
        mask = mask/255.0   # converting mask to (0 and 1) # ----------------- pay attention ------------------ #
        mask = np.expand_dims(mask, axis=-1)  # adding channel axis # ----------------- pay attention ------------------ #
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

"""## Augmentation"""

def get_training_augmentation():
    train_transform = [

        A.OneOf(
            [
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.4),
            ],
            p=0.5,
        ),
        
        A.OneOf(
            [
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=1, border_mode=0), # scale only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=1, border_mode=0), # rotate only
                A.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0), # shift only
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0), # affine transform
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Perspective(p=1),
                A.GaussNoise(p=1),
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.2,
        ),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.2,
        ),
        
    ]

    return A.Compose(train_transform, p=0.9) # 90% augmentation probability


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(512, 512)
    ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

"""## Split dataset"""

#%% Load dataset
x_test_dir = 'dataset/test/images'
y_test_dir = 'dataset/test/labels'

list_IDs_test = os.listdir(x_test_dir)

#%% Parameters
"""## Parameters"""
ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
n_classes = 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001 # learning rate
WEIGHT_DECAY = 1e-5
TO_CATEGORICAL = False
RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)

#%% Enter name of the model that will be loaded
model_name = 'Unet_pscsev1_efficientnet-b7_2023-02-28_10-05-44' #'>>>>>>>>>>>>>>>>Give name<<<<<<<<<<<<<<<<<<<<<<'
print(model_name)

"""# Build model"""

#%%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

'=================================== INFERENCE ================================='
#%%
"""## Inference

Load model
"""
# create segmentation model with pretrained encoder 
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS,    
    classes=n_classes, 
    activation=ACTIVATION,
    decoder_attention_type = 'pscse',
)
   
model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
])
    
# Load model    
checkpoint_loc = 'checkpoints/' + model_name
checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

"""Test dataloader"""
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Test dataloader
test_dataset = Dataset(
    list_IDs_test,
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)

test_dataloader = DataLoader(test_dataset, 
                              batch_size=1, 
                              shuffle=False, 
                              num_workers=2)

"""Evaluation"""
# Loss function
dice_loss = losses.DiceLoss()
focal_loss = losses.FocalLoss() 
total_loss = base.SumOfLosses(dice_loss, focal_loss)

# Evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=total_loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

"""Prediction"""
save_pred = True
threshold = 0.5
ep = 1e-6
raw_pred = []

# Save directory
save_dir_pred = 'predictions/' + model_name
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

# Create dataframe to store records
df = pd.DataFrame(index=[], columns = [
    'Name', 'Accuracy', 'Specificity', 'iou', 'Precision', 'Recall', 'Dice'], dtype='object')

# Create dataframe to store data-based record
df_data = pd.DataFrame(index=[], columns = [
    'Name', 'type', 'Accuracy', 'Specificity', 'iou', 'Precision', 'Recall', 'Dice', 'stp', 'stn', 'sfp', 'sfn'], dtype='object')

# fig, ax = plt.subplots(5,2, figsize=(10,15))
iter_test_dataloader = iter(test_dataloader)

stp, stn, sfp, sfn = 0, 0, 0, 0

for i in range(len(list_IDs_test)):

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension

    image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

    # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])

    pr_mask = model.predict(image.to(DEVICE)) # Move image tensor to gpu

    # Move to CPU and convert to numpy
    gt_mask = gt_mask.squeeze().cpu().numpy()
    pred = pr_mask.squeeze().cpu().numpy()

    # Save raw prediction
    if RAW_PREDICTION: raw_pred.append(pred)

    # Modify prediction based on threshold 
    pred = (pred >= threshold) * 1

    # Save prediction as png
    if save_pred:
        output_im = Image.fromarray((np.squeeze(pred)*255 ).astype(np.uint8))
        output_im.save(os.path.join(save_dir_pred, list_IDs_test[i]))

    # Calculate accuracy, specificity, iou, precision, recall, and dice
    flat_mask = np.squeeze(gt_mask).flatten()
    flat_pred = np.squeeze(pred).flatten()
    
    # Calculate tp, fp, tn, fn
    if np.array_equal(flat_mask, flat_pred): tn, fp, fn, tp = 0, 0, 0, len(flat_mask)
    else: tn, fp, fn, tp = confusion_matrix(flat_mask, flat_pred).ravel()
    
    # Keep adding tp, tn, fp, and fn
    stp += tp
    stn += tn
    sfp += fp
    sfn += fn

    # Calculate metrics
    acc = ((tp + tn)/(tp + tn + fn + fp))*100  
    sp = (tn/(tn + fp + ep))*100
    p = (tp/(tp + fp + ep))*100
    r = (tp/(tp + fn + ep))*100
    # f1 = ((2 * p * r)/(p + r))*100
    dice = (2 * tp / (2 * tp + fp + fn))*100
    iou = (tp/(tp + fp + fn + ep)) * 100
    print("Img # {:1s}, Image {:1s}: acc: {:3f}, sp: {:3f}, iou: {:3f}, p: {:3f}, r: {:3f}, dice: {:3f}".format(str(i+1), name, acc, sp, iou, p, r, dice))

    # Add to dataframe
    tmp = pd.Series([name, acc, sp, iou, p, r, dice], index=['Name', 'Accuracy', 'Specificity', 'iou', 'Precision', 'Recall', 'Dice'])
    df = df.append(tmp, ignore_index = True)
    df.to_csv(os.path.join(save_dir_pred, 'result.csv'), index=False)

print("Mean Accuracy: ", df["Accuracy"].mean())
print("Mean Specificity: ", df["Specificity"].mean())
print('Mean IoU: ', df['iou'].mean())
print("Mean precision: ", df["Precision"].mean())
print("Mean recall: ", df["Recall"].mean())
print("Mean dice: ", df["Dice"].mean())    

raw_pred = np.array(raw_pred)

# Data-based evaluation
sacc = ((stp + stn)/(stp + stn + sfn + sfp))*100  
ssp = (stn/(stn + sfp + ep))*100
siou = (stp/(stp + sfp + sfn + ep))*100
sprecision = (stp/(stp + sfp + ep))*100
srecall = (stp/(stp + sfn + ep))*100
sdice = (2 * stp / (2 * stp + sfp + sfn))*100

print('Data-based accuracy:', sacc)
print('Data-based specificity:', ssp)
print('Data-based iou:', siou)
print('Data-based precision:', sprecision)
print('Data-based recall:', srecall)
print('Data-based dice:', sdice)

tmp2 = pd.Series([name, 'best_model', sacc, ssp, siou, sprecision, srecall, sdice, stp, stn, sfp, sfn], 
                index=['Name', 'type', 'Accuracy', 'Specificity', 'iou', 'Precision', 'Recall', 'Dice', 'stp', 'stn', 'sfp', 'sfn'])
df_data = df_data.append(tmp2, ignore_index = True)


df_data.to_csv(os.path.join('predictions',  model_name + '_data_based_result.csv'), index=False)


# Save raw prediction in .mat format
if RAW_PREDICTION:
  raw_pred = np.array(raw_pred)
  sio.savemat(os.path.join(save_dir_pred, 'raw_pred.mat'), {'p': raw_pred}, do_compression=True)
