import numpy as np
import pandas as pd
import pydicom as pdc
import re, random
import torch, os
from torch.utils.data import Dataset
from typing import Optional, List

# INDEXING
def getPaths(path: str):
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

def createImageIndexCSV(path:str, force=False, result_file_path = "data_indices.csv"):
    if os.path.isfile(result_file_path) and not force:
        return pd.read_csv(result_file_path, delimiter=',', index_col=0)
    result_frame = []
    paths, dirs = getPaths(path)
    for patient in dirs:
        patient_data = []
        slice_positions = []
        patient_paths = get_list_of_paths(paths, patient)
        for image_path in patient_paths:
            try:
                dcm = pdc.read_file(image_path)
                patient_data.append([patient, int(dcm.AcquisitionNumber), dcm.SliceLocation, image_path])
                slice_positions.append(dcm.SliceLocation)
            except:
                continue
        slice_positions = np.unique(np.array(slice_positions))
        for data in patient_data:
            slice_num = np.where(slice_positions == data[2])
            result_frame.append([data[0], data[1], int(slice_num[0]) + 1, data[3]])
    df = pd.DataFrame(result_frame, columns= ["Patient", "Volume", "Slice", "ImagePath"])
    df.to_csv(result_file_path)
    return df

def createAIFLabelIndexCSV(path:str, force=False):
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

# SPLITTING
def split_patients_to_txt(data: pd.DataFrame, seed: int, distribution: tuple = (80, 10, 10)):
    if np.sum(distribution) != 100: raise Exception(f"The distribution {distribution} does not sum to 100%!")
    train_size, val_size, test_size = distribution[0]/100, distribution[1]/100, distribution[2]/100
    np.random.seed(seed)
    patients = np.unique(data['Patient'])
    train_selection = np.random.choice(patients, size = int(len(patients)*train_size), replace=False)
    patients = patients[~np.isin(patients, train_selection)]
    val_selection = np.random.choice(patients, size = int(len(patients)*(val_size/(val_size+test_size))), replace=False)
    test_selection = patients[~np.isin(patients, val_selection)]
    with open('train_patients.txt', 'w') as f:
        for i, line in enumerate(train_selection):
            if i != len(train_selection)-1:
                f.write(f"{line},")
            else:
                f.write(f"{line}")
    with open('val_patients.txt', 'w') as f:
        for i, line in enumerate(val_selection):
            if i != len(val_selection)-1:
                f.write(f"{line},")
            else:
                f.write(f"{line}")
    with open('test_patients.txt', 'w') as f:
        for i, line in enumerate(test_selection):
            if i != len(test_selection)-1:
                f.write(f"{line},")
            else:
                f.write(f"{line}")

def get_patients_from_txt(paths = {"train" : "train_patients.txt", "val" : "val_patients.txt", "test" : "test_patients.txt"}):
    patients = {}
    for name in paths.keys():
        array = np.loadtxt(paths.get(name), dtype=int, delimiter=",", unpack=False)
        patients[name] = array
    return patients

def sort_imagedata_on_patients(data: pd.DataFrame, patient_distribution: dict):
    train_patients = patient_distribution.get("train")
    val_patients = patient_distribution.get("val")
    test_patients = patient_distribution.get("test")

    train_mask = np.isin(data["Patient"], train_patients)
    val_mask = np.isin(data["Patient"], val_patients)
    test_mask = np.isin(data["Patient"], test_patients)

    return data[train_mask], data[val_mask], data[test_mask]


def get_slice_volumes_for_one_patient(data:pd.DataFrame, patient:int, slice_number:int):
    slices = data.loc[(data['Patient'] == patient) & (data['Slice'] == slice_number)]
    return slices

def create_cross_val_splits(image_data: pd.DataFrame, SEED:int, num_splits:int=5):
    patients = np.unique(image_data['Patient'])
    patients_per = len(patients) // num_splits
    np.random.seed(SEED)
    clusters = {}
    for c in range(num_splits):
        selection = np.random.choice(patients, size=patients_per, replace=False)
        patients = patients[~np.isin(patients, selection)]
        selection_mask = np.isin(image_data['Patient'], selection)
        clusters[c] =  image_data[selection_mask]
    
    i = 0
    while len(patients) > 0:
        selection = np.random.choice(patients, size=1)
        patients = patients[~np.isin(patients, selection)]
        selection_mask = np.isin(image_data['Patient'], selection)
        clusters[i] = pd.concat([clusters[i], image_data[selection_mask]], axis=0)
        if i == num_splits-1: i = 0
        else: i += 1
    return clusters

# PREPROCESSING
def createImageArray(paths: List[str]):
    base_slice = pdc.read_file(paths.iloc[[0]]['ImagePath'].item())
    base_slice = base_slice.pixel_array
    slices = np.zeros((np.max(paths['Volume']), np.max(paths['Slice']), base_slice.shape[0], base_slice.shape[1]))
    for volume_num in range(1, slices.shape[0]+1):
        for slice_num in range(1, slices.shape[1]+1):
            path = paths.loc[(paths['Volume'] == volume_num) & (paths['Slice'] == slice_num)]
            image = pdc.read_file(path['ImagePath'].item())
            slices[volume_num-1, slice_num-1, :, :] = image.pixel_array
    return slices

def get_max_intensity_volume_index(array: np.ndarray):
    vol_intensities = np.zeros(array.shape[0])
    slopes = np.zeros(array.shape[0])
    for volume_num in range(array.shape[0]):
        vol_intensities[volume_num] = np.sum(array[volume_num, : , : , :])
    for i in range(len(vol_intensities)-1):
        slopes[i] = vol_intensities[i+1] - vol_intensities[i]
    return np.argmax(slopes) +1

def get_max_intensity_for_dataset(data: pd.DataFrame, force = False):
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

def add_extra_volumes(image_data:pd.DataFrame, labels:pd.DataFrame, copy_data:pd.DataFrame, extra_vols:int, patient:int, volume_index:int):
    # Adds extra volumes of positively labeled slices
    extra_count = 0
    for i in range(int(volume_index)+1, extra_vols+int(volume_index)+1):
        extra_data = image_data[(image_data['Patient']== patient) & (image_data['Slice'].isin(labels[(labels['Patient'] == patient) & (labels['Label'] == 1)]['Slice']))]
        extra_count += len(extra_data[extra_data['Volume'] == i])
        copy_data = pd.concat([copy_data, extra_data[extra_data['Volume'] == i]])
    return copy_data, extra_count

def filter_on_intensity_and_add_data(image_data:pd.DataFrame, labels:pd.DataFrame, vol_intensities:np.ndarray, extra_vols=0):
    copy_data = image_data.copy()
    original_count = len(copy_data)
    extra_count = 0
    patient_filter = np.isin(vol_intensities[:,0], np.unique(copy_data['Patient']))
    vol_intensities = vol_intensities[patient_filter]
    for patient, volume_index in zip(vol_intensities[:, 0], vol_intensities[:, 1]):
        before = len(copy_data)
        copy_data = copy_data.drop(copy_data[copy_data['Patient'] == int(patient)].index.intersection(copy_data[copy_data['Volume'] != int(volume_index)].index))
        original_count -= before - len(copy_data)

        if extra_vols + volume_index < 80:
            copy_data, count = add_extra_volumes(image_data, labels, copy_data, extra_vols, patient, volume_index)
            extra_count += count
    if extra_vols > 0:
        print(f"Added data {extra_count}. Original amount of data {original_count}")
    return copy_data

def crop_image(image: np.ndarray, reduction: float = 0.70):
    y_bottom, y_top = int(image.shape[0]*reduction), int(image.shape[0]*(1-reduction))
    return image[y_top:y_bottom, y_top:y_bottom]

def load_and_prepare_images_from_txt(image_data:pd.DataFrame, labels:pd.DataFrame, vol_intensities:np.ndarray, SEED:int, extra_vols:int = 0, permute:bool = False):
    
    copy_data = filter_on_intensity_and_add_data(image_data, labels, vol_intensities, extra_vols)
    dist = get_patients_from_txt()
    train_images, val_images, test_images = sort_imagedata_on_patients(copy_data, dist)

    train_labels = np.isin(labels['Patient'], np.unique(train_images['Patient']))
    train_labels = labels[train_labels]
    val_labels = np.isin(labels['Patient'], np.unique(val_images['Patient']))
    val_labels = labels[val_labels]
    test_labels = np.isin(labels['Patient'], np.unique(test_images['Patient']))
    test_labels = labels[test_labels]

    if permute:
        train_labels['Label'] = train_labels['Label'].sample(frac=1, random_state=SEED).values
    
    label_train_true_size, label_train_size = len(train_labels[train_labels['Label'] == 1]) + len(train_images) - len(train_labels), len(train_images)
    label_val_true_size, label_val_size = len(val_labels[val_labels['Label'] == 1]) + len(val_images)  - len(val_labels), len(val_images)
    label_test_true_size, label_test_size = len(test_labels[test_labels['Label'] == 1]) + len(test_images) - len(test_labels), len(test_images)
    print(f"Size training, val, and test data: {len(train_images)}, {len(val_images)}, {len(test_images)}.")
    print(f"True labels in training: {label_train_true_size} of {label_train_size} ({round(label_train_true_size/label_train_size, 3)*100:.1f}%).")
    print(f"True labels in validation: {label_val_true_size} of {label_val_size} ({round(label_val_true_size/label_val_size, 3)*100:.1f}%).")
    print(f"True labels in test: {label_test_true_size} of {label_test_size} ({round(label_test_true_size/label_test_size, 3)*100:.1f}%).")

    return train_images, val_images, test_images, train_labels, val_labels, test_labels


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

    def __init__(self, X_df: pd.DataFrame, y_df: pd.DataFrame, normalize: bool = False, crop: Optional[float] = None):
        self.X = X_df
        self.y = y_df
        self.normalize = normalize
        self.crop = crop
    
    def __len__(self):
        return(len(self.X))
    
    def __getitem__(self, index:int):
        selected_X = self.X.iloc[[index]]
        image = pdc.read_file(selected_X["ImagePath"].item()).pixel_array.astype('int16')
        if self.crop is not None:
            image = crop_image(image, self.crop)
        if self.normalize:
            image = image / np.max(image)
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0)
        label = self.y[(self.y['Patient'] == int(selected_X['Patient'])) & (self.y['Slice'] == int(selected_X['Slice']))]['Label'].item()
        label = torch.tensor(label, dtype=torch.float)
        return image, label