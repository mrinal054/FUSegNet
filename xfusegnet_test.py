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
model_name = 'Unet_scse_efficientnet-b7_2023-01-20_20-13-27' #'>>>>>>>>>>>>>>>>Give name<<<<<<<<<<<<<<<<<<<<<<'
print(model_name)

"""# Build model"""

#%%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

'=================================== INFERENCE ================================='
#%% Inference
from PIL import Image

RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)
save_pred = True
verbose = False
threshold = 0.5
ep = 1e-6

# Get checkpoint directory
checkpoint_loc = 'checkpoints/' + model_name

with open(os.path.join(checkpoint_loc, model_name + '.pickle'), 'rb') as f:
    store_kfold_best_log = pickle.load(f)

get_val_loss = []
get_epochs = []    
for row in store_kfold_best_log:
    get_val_loss.append(row[1])
    get_epochs.append(row[2])
    
min_val_loss_idx = np.argmin(get_val_loss)
max_val_loss_idx = np.argmax(get_val_loss)

max_epoch = np.max(get_epochs)
avg_epoch = int(np.mean(get_epochs))

ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'

# Model names
models = [
    'best_model_1.pth',
    'best_model_2.pth',
    'best_model_3.pth',
    'best_model_4.pth',
    'best_model_5.pth',     
    ]
                                       
'===================================== VOTING/AVG ========================================='
#%% Prediction based on voting
store_pred = []

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

print('\n')
for m_name in models:
    
    print(f"Generating prediction for {m_name}")

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
    checkpoint = torch.load(os.path.join(checkpoint_loc, m_name))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    iter_test_dataloader = iter(test_dataloader)
    
    raw_pred = []
    
    for i in range(len(list_IDs_test)):
    
        name = os.path.splitext(list_IDs_test[i])[0] # remove extension
    
        image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors
    
        # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])
    
        pr_mask = model.predict(image.to(DEVICE)) # Move image tensor to gpu
    
        # Move to CPU and convert to numpy        
        pred = pr_mask.squeeze().cpu().numpy()
    
        # Save raw prediction
        raw_pred.append(pred)
        
    store_pred.append(raw_pred)


'--------------------------------- Average ------------------------------------'
#%% Average
print('\n\nGenerating output taking average of all raw predictions')

# Save directory
save_dir_pred = 'predictions/' + model_name + '_avg'
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

iter_test_dataloader = iter(test_dataloader)    
    
for i in range(len(list_IDs_test)):

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension

    image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

    # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])
        
    temp_store = []
    for j in range(len(store_pred)): # iterate over all models
        curr_pred = store_pred[j][i] # get jth model's prediction for the current image
        temp_store.append(curr_pred)
        
    temp_store = np.array(temp_store)  
    
    temp_store = np.mean(temp_store, axis=0) # take mean along axis=0
    
    pred = (temp_store >= threshold) * 1
    
    gt_mask = gt_mask.squeeze().cpu().numpy()
    
    # Save prediction as png
    if save_pred:
        output_im = Image.fromarray((np.squeeze(pred)*255 ).astype(np.uint8))
        output_im.save(os.path.join(save_dir_pred, list_IDs_test[i]))

'--------------------------------- voting ------------------------------------'
#%% Voting
print('\n\nGenerating output based on voting.')

# Note: voting is easier if there are odd no. of models.
# So, if there are even no. of models, we will remove that k-fold model which has max val loss
if len(store_pred) % 2 == 0: 
    print(f"Found even no. of models for voting. So, removing best_model_{max_val_loss_idx} from k-fold models")
    # store_pred.remove(store_pred[max_val_loss_idx + 1]) # adding one because store_pred has prediction for last_model
                                            # which is not in k-fold. 
    store_pred_temp = []
    for pc in range(len(store_pred)):
        if pc != max_val_loss_idx + 1:
            store_pred_temp.append(store_pred[pc])
    del store_pred
    store_pred = store_pred_temp
    del store_pred_temp
            
threshold_sum = int(np.ceil(len(store_pred)/2)) # considering odd numbers of models
# Note: If no. of models = 5, then threshold_sum = 3. 
# So, consider for binary segmentation,
# if sum of the values of a pixel located at (x,y) position for all 5 prediction is 3, it means
# that three predictions voted 1 and two predictions voted 0. So, 1 is winner.
# Finally, the value of that pixel at (x,y) will be set to 1.

# Save directory
save_dir_pred = 'predictions/' + model_name + '_voting'
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

iter_test_dataloader = iter(test_dataloader)    
    
for i in range(len(list_IDs_test)):

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension

    image, gt_mask = next(iter_test_dataloader) # get image and mask as Tensors

    # Note: Image shape: torch.Size([1, 3, 512, 512]) and mask shape: torch.Size([1, 1, 512, 512])
        
    temp_store = []
    for j in range(len(store_pred)): # iterate over all models
        curr_pred = store_pred[j][i] # get jth model's prediction for the current image
        curr_pred = (curr_pred >= threshold) * 1 # converting to binary image
        temp_store.append(curr_pred)
        
    temp_store = np.array(temp_store)
        
    temp_store = np.sum(temp_store, axis=0) # take sum along axis=0
    
    pred = (temp_store >= threshold_sum) * 1
    
    gt_mask = gt_mask.squeeze().cpu().numpy()
    
    # Save prediction as png
    if save_pred:
        output_im = Image.fromarray((np.squeeze(pred)*255 ).astype(np.uint8))
        output_im.save(os.path.join(save_dir_pred, list_IDs_test[i]))