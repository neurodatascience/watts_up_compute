import nibabel as nib
import numpy as np
import pandas as pd
from dask import compute, delayed
import dask.multiprocessing

def convert_mgz_to_nifti(mgz_file):
    nii_file = mgz_file.rsplit('.',1)[0] + '.nii.gz'
    mgh = nib.load(mgz_file)
    nib.save(mgh, nii_file)

def dice_coef(y_true, y_pred, smooth=1):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = np.mean((2. * intersection + smooth)/(union + smooth))
    return dice

def get_multi_label_dice(label_list, y_true, y_pred):
    ''' Calculates label-wise dice score for a 3D segmentation volume
    '''
    # # Dask based parallel compute
    values = [delayed(dice_coef)(y_true == label, y_pred == label) 
              for label in label_list]

    dice_list = compute(*values, scheduler='threads',num_workers=2) 
    dice_dict = dict(zip(label_list,dice_list)) 

    return pd.DataFrame(list(dice_dict.items()),columns=['label','dice'])