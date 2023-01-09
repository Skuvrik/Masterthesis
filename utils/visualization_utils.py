import matplotlib.pyplot as plt
import random as rnd
import pydicom as pdc

def display_predictions(sorted_data):
    fig, axs = plt.subplots(3, 4, figsize=(10, 13))
    for i, ax in enumerate(axs):
        TP_image = rnd.choice(sorted_data[0])['ImagePath']
        ax[0].imshow(pdc.read_file(TP_image).pixel_array, cmap='viridis')
        ax[0].grid(False)
        ax[0].axis('off')
        FP_image = rnd.choice(sorted_data[1])['ImagePath']
        ax[1].imshow(pdc.read_file(FP_image).pixel_array, cmap='viridis')
        ax[1].grid(False)
        ax[1].axis('off')
        TN_image = rnd.choice(sorted_data[2])['ImagePath']
        ax[2].imshow(pdc.read_file(TN_image).pixel_array, cmap='viridis')
        ax[2].grid(False)
        ax[2].axis('off')
        FN_image = rnd.choice(sorted_data[3])['ImagePath']
        ax[3].imshow(pdc.read_file(FN_image).pixel_array, cmap='viridis')
        ax[3].grid(False)
        ax[3].axis('off')
        if i == 0:
            ax[0].set_title('True positive')
            ax[1].set_title('False positive')
            ax[2].set_title('True  negative')
            ax[3].set_title('False negative')
    fig.tight_layout()