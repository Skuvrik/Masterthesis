import matplotlib.pyplot as plt
import random as rnd
import pydicom as pdc
from captum.attr import Occlusion
import numpy as np
import torch
from utils.datahandling_utils import crop_image
from typing import Optional


def get_model_input(slice: object, device: str, normalize: bool, crop: Optional[float] = None):
    image  = pdc.read_file(slice['ImagePath']).pixel_array
    if crop is not None:
        image = crop_image(image, crop)
    if normalize:
        image = image / np.max(image)
    image = np.expand_dims(image, axis=(0,1))
    image = torch.tensor(image, dtype=torch.float).to(device)
    return image

def analyze_occlusion(image: torch.tensor, model: object):
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(image, strides=(1,5,5), 
                                            target=None, 
                                            sliding_window_shapes=(1,6,6),
                                            baselines=0)
    return attributions_occ

def display_predictions(sorted_data: list, model: object = None, device: str = None, occlude: bool = False, normalize: bool = False, crop: Optional[float] = None):
    TP, FP, TN, FN = np.minimum(4,len(sorted_data[0])), np.minimum(4,len(sorted_data[1])), np.minimum(4,len(sorted_data[2])), np.minimum(4,len(sorted_data[3]))
    number_array = np.array([TP, FP, TN, FN])
    rows = np.min(number_array[np.nonzero(number_array)])
    cols = np.count_nonzero(number_array)
    count = 0
    fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
    for i, ax in enumerate(axs):
        if TP > 0:
            if i == 0: 
                TP_index = count
                count+=1
            TP_slice = rnd.choice(sorted_data[0])
            TP_image = get_model_input(TP_slice, device, normalize, crop)
            ax[TP_index].imshow(TP_image.squeeze().cpu().detach().numpy(), cmap='gray')
            if occlude:
                TP_occlusion = analyze_occlusion(TP_image, model)
                ax[TP_index].imshow(TP_occlusion.squeeze().cpu().detach().numpy(), cmap='cool', alpha=.5)
            ax[TP_index].grid(False)
            ax[TP_index].axis('off')
            
        if FP > 0:
            if i == 0: 
                FP_index = count
                count+=1
            FP_slice = rnd.choice(sorted_data[1])
            FP_image = get_model_input(FP_slice, device, normalize, crop)
            ax[FP_index].imshow(FP_image.squeeze().cpu().detach().numpy(), cmap='gray')
            if occlude:
                FP_occlusion = analyze_occlusion(FP_image, model) 
                ax[FP_index].imshow(FP_occlusion.squeeze().cpu().detach().numpy(), cmap='cool', alpha=.5)
            ax[FP_index].grid(False)
            ax[FP_index].axis('off')
            
        if TN > 0:
            if i == 0: 
                TN_index = count
                count+=1
            TN_slice = rnd.choice(sorted_data[2])
            TN_image = get_model_input(TN_slice, device, normalize, crop)
            ax[TN_index].imshow(TN_image.squeeze().cpu().detach().numpy(), cmap='gray')
            if occlude:
                TN_occlusion = analyze_occlusion(TN_image, model)
                ax[TN_index].imshow(TN_occlusion.squeeze().cpu().detach().numpy(), cmap='cool', alpha=.5)
            ax[TN_index].grid(False)
            ax[TN_index].axis('off')
            
        if FN > 0:
            if i == 0:
                FN_index = count
            FN_slice = rnd.choice(sorted_data[3])
            FN_image = get_model_input(FN_slice, device, normalize, crop)
            ax[FN_index].imshow(FN_image.squeeze().cpu().detach().numpy(), cmap='gray')
            if occlude:
                FN_occlusion = analyze_occlusion(FN_image, model)
                ax[FN_index].imshow(FN_occlusion.squeeze().cpu().detach().numpy(), cmap='cool', alpha=.5)
            ax[FN_index].grid(False)
            ax[FN_index].axis('off')

        if i == 0:
            if TP > 0: ax[TN_index].set_title('True positive')
            if FP > 0: ax[FP_index].set_title('False positive')
            if TN > 0: ax[TN_index].set_title('True negative')
            if FN > 0: ax[FN_index].set_title('False negative')
    fig.tight_layout()