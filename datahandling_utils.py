import numpy as np
import pandas as pd
import pydicom as pdc
import re, random
import torch, os
from torch.utils.data import Dataset

# INDEXING
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
    if len(dir_list) < 1:
        return paths, dir_list
    return paths, dir_list[0]

def createImageIndexCSV(path, force=False):
    result_file_path = "data_indices.csv"
    if os.path.isfile(result_file_path) and not force:
        return pd.read_csv(result_file_path, delimiter=',', index_col=0)
    result_frame = []
    paths, dirs = getPaths(path)
    for patient in dirs:
        patient_paths = get_list_of_paths(paths, patient)
        for image in patient_paths:
            slice_num, volume_num = get_curr_slice_and_volume(image)
            result_frame.append([patient, volume_num, slice_num, image])
    df = pd.DataFrame(result_frame, columns= ["Patient", "Volume", "Slice", "ImagePath"])
    df.to_csv(result_file_path)
    return df

def createAIFLabelIndexCSV(path, force=False):
    result_path = "label_indices.csv"
    if os.path.isfile(result_path) and not force:
        return pd.read_csv(result_path, delimiter=',', index_col=0)
    paths, _ = getPaths(path)
    label_indexes = [((3-len(str(x)))*"0")+str(x) for x in range(1, len(paths))]
    result_frame = []
    for index, path in zip(label_indexes, paths):
        result_frame.append([index, path])
    df = pd.DataFrame(result_frame, columns=["Index", "LabelPath"])
    df.to_csv(result_path)
    return df

def get_list_of_paths(paths, patient):
    patient_filter = re.compile("(.*?)\/" + patient + "\/.*")
    patient_paths = list(filter(patient_filter.match, paths))
    return patient_paths

def get_curr_slice_and_volume(image, num_slices = 48):
    split_list = image.split("_")
    number = int(split_list[-1].split(".")[0])
    if (number <= num_slices):
        slice_num = number % (num_slices + 1)
        volume_num = number // (num_slices + 1)
    elif number % num_slices == 0:
        slice_num = num_slices
        volume_num = (number // num_slices) - 1
    else:
        slice_num = number % num_slices
        volume_num = number // num_slices 
    return slice_num, volume_num+1

# SPLITTING
def get_train_test_split_on_patients(data, train_size = 0.8, slice_number=None):
    # Jezuz this needs more consistent datastructures.
    random.seed(42)
    patient_num = np.max(data['Patient'])
    selection = random.sample(list(range(1, patient_num+1)), k = patient_num)
    train_selection = random.sample(selection, k = int(patient_num*train_size))
    test_mask = np.isin(selection, train_selection)
    test_selection = np.array(selection)[~test_mask]
    test_selection = random.sample(list(test_selection), k = patient_num-len(train_selection))

    test_mask = np.isin(data['Patient'], test_selection)
    train_mask = np.isin(data['Patient'], train_selection)
    train = data[train_mask]
    test = data[test_mask]
    if slice_number is not None:
        train = train.loc[train['Slice'] == slice_number]
        test = test.loc[test['Slice'] == slice_number]
    return train, test

def get_slice_volumes_for_one_patient(data, patient, slice_number):
    slices = data.loc[(data['Patient'] == patient) & (data['Slice'] == slice_number)]
    return slices

# PREPROCESSING
def createImageArray(paths):
    base_slice = pdc.read_file(paths.iloc[[0]]['ImagePath'].item())
    base_slice = base_slice.pixel_array
    slices = np.zeros((np.max(paths['Volume']), np.max(paths['Slice']), base_slice.shape[0], base_slice.shape[1]))
    for volume_num in range(1, slices.shape[0]+1):
        for slice_num in range(1, slices.shape[1]+1):
            path = paths.loc[(paths['Volume'] == volume_num) & (paths['Slice'] == slice_num)]
            image = pdc.read_file(path['ImagePath'].item())
            slices[volume_num-1, slice_num-1, :, :] = image.pixel_array
    return slices

def get_max_intensity_volume_index(array):
    vol_intensities = np.zeros(array.shape[0])
    slopes = np.zeros(array.shape[0])
    for volume_num in range(array.shape[0]):
        vol_intensities[volume_num] = np.sum(array[volume_num, : , : , :])
    for i in range(len(vol_intensities)-1):
        slopes[i] = vol_intensities[i+1] - vol_intensities[i]
    return np.argmax(slopes) +1

def get_max_intensity_for_dataset(data, force = False):
    patients = np.unique(data['Patient'])
    path = 'volume_intensities.csv'
    if os.path.isfile(path) and not force:
        intensities = np.genfromtxt(path, delimiter=',')
        if np.in1d(patients, intensities[:,0]).all():
            return intensities
    intensities = np.zeros((len(patients), 2))
    for i, patient in enumerate(patients):
        paths = data.loc[(data['Patient'] == patient)]
        image_array = createImageArray(paths)
        intensities[i, 0] = patient
        intensities[i, 1] = get_max_intensity_volume_index(image_array)
    np.savetxt(path, intensities, delimiter=',')
    return intensities

def add_extra_volumes(image_data, labels, copy_data, extra_vols, patient, volume_index):
    # Adds extra volumes of positively labeled slices
    extra_count = 0
    for i in range(int(volume_index)+1, extra_vols+int(volume_index)+1):
        extra_data = image_data[(image_data['Patient']== patient) & (image_data['Slice'].isin(labels[(labels['Patient'] == patient) & (labels['Label'] == 1)]['Slice']))]
        extra_count += len(extra_data[extra_data['Volume'] == i])
        copy_data = pd.concat([copy_data, extra_data[extra_data['Volume'] == i]])
    return copy_data, extra_count

def filter_on_intensity_and_add_data(image_data, labels, vol_intensities, extra_vols=0):
    copy_data = image_data.copy()
    original_count = len(copy_data)
    extra_count = 0
    patient_filter = np.isin(vol_intensities[:,0], np.unique(copy_data['Patient']))
    vol_intensities = vol_intensities[patient_filter]
    for patient, volume_index in zip(vol_intensities[:, 0], vol_intensities[:, 1]):
        before = len(copy_data)
        copy_data = copy_data.drop(copy_data[copy_data['Patient'] == int(patient)].index.intersection(copy_data[copy_data['Volume'] != int(volume_index)].index))
        original_count -= before - len(copy_data)

        if extra_vols + volume_index < 48:
            copy_data, count = add_extra_volumes(image_data, labels, copy_data, extra_vols, patient, volume_index)
            extra_count += count
    if extra_vols > 0:
        print(f"Added data {extra_count}. Original amount of data {original_count}")
    return copy_data

# DATASETS
class ImageDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X = X_df
        self.y = y_df
    
    def __len__(self):
        return(len(self.X))
    
    def __getitem__(self, index):
        dcm = pdc.read_file(self.X.iloc[[index]]["ImagePath"].item())
        image = dcm.pixel_array.astype("int32")
        label = np.loadtxt(self.y['LabelPath'][index])

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)

class SliceIntensityDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X = X_df
        self.y = y_df
    
    def __len__(self):
        return(len(self.X))
    
    def __getitem__(self, index):
        selected_X = self.X.iloc[[index]]
        dcm = pdc.read_file(selected_X["ImagePath"].item())
        image = dcm.pixel_array.astype('int16')
        label = self.y[(self.y['Patient'] == int(selected_X['Patient'])) & (self.y['Slice'] == int(selected_X['Slice']))]['Label'].item()
        image = image / np.max(image)
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)

# Maybe useful code..
# train, test = get_train_test_split_on_patients(data, 5, slice_number = 20)
# print(f"Train size: {train.size} Test size: {test.size}")
# train_data, test_data = ImageDataset(train, np.zeros(len(train))), ImageDataset(test, np.zeros(len(test)))
# train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, pin_memory=False)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, pin_memory = False)
