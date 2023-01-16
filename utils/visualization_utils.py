import matplotlib.pyplot as plt
import random as rnd
import pydicom as pdc
from captum.attr import Occlusion
import numpy as np
import torch

def get_model_input(slice: object, device: str):
    image  = pdc.read_file(slice['ImagePath']).pixel_array
    image = image.astype('int16')
    image = np.expand_dims(image, axis=(0,1))
    image = image / np.max(image)
    image = torch.tensor(image, dtype=torch.float).to(device)
    return image

def analyze_occlusion(image: torch.tensor, model: object):
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(image, strides=(1,20,20), 
                                            target=None, 
                                            sliding_window_shapes=(1,30,30),
                                            baselines=0)
    return attributions_occ

def display_predictions(sorted_data: list, model: object = None, device: str = None, occlude: bool = False):
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))
    for i, ax in enumerate(axs):
        TP_slice = rnd.choice(sorted_data[0])
        TP_image = get_model_input(TP_slice, device)
        ax[0].imshow(TP_image.squeeze().cpu().detach().numpy(), cmap='gray')
        if occlude:
            TP_occlusion = analyze_occlusion(TP_image, model)
            ax[0].imshow(TP_occlusion.squeeze().cpu().detach().numpy(), cmap='viridis', alpha=.5)
        ax[0].grid(False)
        ax[0].axis('off')

        FP_slice = rnd.choice(sorted_data[1])
        FP_image = get_model_input(FP_slice, device)
        ax[1].imshow(FP_image.squeeze().cpu().detach().numpy(), cmap='gray')
        if occlude:
            FP_occlusion = analyze_occlusion(FP_image, model) 
            ax[1].imshow(FP_occlusion.squeeze().cpu().detach().numpy(), cmap='viridis', alpha=.5)
        ax[1].grid(False)
        ax[1].axis('off')

        TN_slice = rnd.choice(sorted_data[2])
        TN_image = get_model_input(TN_slice, device)
        ax[2].imshow(TN_image.squeeze().cpu().detach().numpy(), cmap='gray')
        if occlude:
            TN_occlusion = analyze_occlusion(TN_image, model)
            ax[2].imshow(TN_occlusion.squeeze().cpu().detach().numpy(), cmap='viridis', alpha=.5)
        ax[2].grid(False)
        ax[2].axis('off')
        
        FN_slice = rnd.choice(sorted_data[3])
        FN_image = get_model_input(FN_slice, device)
        ax[3].imshow(FN_image.squeeze().cpu().detach().numpy(), cmap='gray')
        if occlude:
            FN_occlusion = analyze_occlusion(FN_image, model)
            ax[3].imshow(FN_occlusion.squeeze().cpu().detach().numpy(), cmap='viridis', alpha=.5)
        ax[3].grid(False)
        ax[3].axis('off')
        if i == 0:
            ax[0].set_title('True positive')
            ax[1].set_title('False positive')
            ax[2].set_title('True  negative')
            ax[3].set_title('False negative')
    fig.tight_layout()