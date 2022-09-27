import numpy as np
import pandas as pd
import pydicom as pdc
import re
from pydicom.sequence import Sequence
import torch, os
from torch.utils.data import Dataset

def getPaths(path):
    paths = []
    dir_list = []
    for root, dirs, files in os.walk(path):
        if len(dirs) < 1:
            for i in range(len(files)):
                path = '/'
                path_list = (root+'/'+files[i]).split('\\')
                path = path.join(path_list)
                paths.append(path)
        else:
            dir_list.append(dirs)
    return paths, dir_list

def createImageIndex(path, force=False):
    result_file_path = "data_indices.csv"
    if os.path.isfile(result_file_path) and not force:
        return
    result_frame = []
    paths, dirs = getPaths(path)
    for patient in dirs:
        reg_filter = re.compile("(.*?)\/" + patient + "\/.*")
        patient_paths = list(filter(reg_filter.match, paths))
        num_volumes = patient_paths[-1].split("_")[0]
        for image in patient_paths:
            split_list = image.split("_")
            slice_num = split_list[-1] % 48
            volume_num = split_list[-1] % num_volumes
            result_frame.append([patient, volume_num, slice_num, image])
    df = pd.DataFrame(result_frame, index = False)
    df.to_csv(result_file_path)

class ImageDataset(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels
    
    def __len__(self):
        return(len(self.X))
    
    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)

if __name__ == "__main__":

    createImageIndex("D:/iCAT_IMAGES")
