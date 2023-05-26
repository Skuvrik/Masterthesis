from operator import itemgetter
import torch
from torch import nn
import numpy as np
import pydicom as pdc
from typing import Optional, Callable, List
from models.mouridsen_new import mouridsen2006
from utils.datahandling_utils import create_image_array, crop_image, get_image_volume
from utils.signal_conversion import pass_through

def train(train_loader, model, criterion, optimizer, device:str):
    losses = 0
    torch.cuda.empty_cache()
    model.train()
    for i, (input,target) in enumerate(train_loader):
        target = torch.unsqueeze(target,1)
        target = target.to(device)
        input = input.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss / len(target)
    return losses

def validate(test_loader, model, criterion, device:str, apr: bool = False):
    torch.cuda.empty_cache()
    model.eval()
    losses = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = torch.unsqueeze(target,1)
            target = target.to(device)
            input = input.to(device)
            output = model(input)
            if apr:
                for actual, predicted in zip(target, output):
                    predicted = int(round(torch.sigmoid(predicted).item()))
                    actual = int(actual.item())
                    if predicted == 1 and actual == 1:
                        TP += 1
                    elif predicted == 1 and actual == 0:
                        FP += 1
                    elif predicted == 0 and actual == 0:
                        TN += 1
                    elif predicted == 0 and actual == 1:
                        FN += 1
                    else:
                        pass
            loss = criterion(output, target)
            losses += loss / len(target)
        if apr:
            try:
                accuracy = round((TP+TN)/(TP+FP+TN+FN),2)
            except:
                accuracy = 0
            try:
                precision = round(TP/(FP+TP), 2)
            except:
                precision = 0
            try: 
                recall = round(TP/(TP+FN), 2)
            except:
                recall = 0
            return [losses, accuracy, precision, recall]
    return [losses]

def predict(loader, model, device:str):
    predictions = []
    labels = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (input, target) in loader:
            target = torch.unsqueeze(target,1)
            target = target.to(device)
            input = input.to(device)
            output = torch.sigmoid(model(input))
            
            predictions.extend(output.numpy())
            labels.extend(target.numpy())
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    return np.concatenate((predictions, labels), axis=1)

def train_and_eval(model, train_loader, test_loader, optimizer, criterion, epochs, device, apr=True):
    losses = np.zeros((epochs, 2))
    metrics = np.zeros((epochs, 3))
    model = model.to(device)
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer, device)
        val_loss = validate(test_loader, model, criterion, device, apr)
        losses[epoch,0], losses[epoch,1] = train_loss.item(), val_loss[0].item()

        if apr:
            metrics[epoch,0], metrics[epoch,1], metrics[epoch,2] = val_loss[1], val_loss[2], val_loss[3]

        if (epoch+1) % 5 == 0 or epoch+1 == epochs:
            print(f"Epoch {epoch + 1} of {epochs}")
            if apr:
                print(f"Validation - Accuracy {metrics[epoch,0]} Precision {metrics[epoch, 1]} and Recall {metrics[epoch, 2]}")
            print(f"Training loss {losses[epoch, 0]}, validation loss {losses[epoch,1]}\n")

    return [losses, metrics]

def get_model_performance_metrics(model, images, labels, device, normalize, crop: Optional[float] = None):
    # Copy dataframe due to the use of iterrows
    images_copy = images.copy()
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    for _, selected in images_copy.iterrows():
        model.eval()
        image = pdc.read_file(selected['ImagePath']).pixel_array
        if crop is not None:
            image = crop_image(image, crop)
        if normalize:
            image = image / np.max(image)
        image = np.expand_dims(image, axis=(0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.to(device)
        actual = labels[(labels['Patient'] == int(selected['Patient'])) & ((
            labels['Slice'] == int(selected['Slice'])) & (
            labels['Series'] == int(selected['Series'])))]['Label'].item()
        predicted = int(round(torch.sigmoid(model(image)).item()))
        if predicted == 1 and actual == 1:
            TP_list.append(selected)
        elif predicted == 1 and actual == 0:
            FP_list.append(selected)
        elif predicted == 0 and actual == 0:
            TN_list.append(selected)
        elif predicted == 0 and actual == 1:
            FN_list.append(selected)
        else:
            pass
    return [TP_list, FP_list, TN_list, FN_list]

### Specific for KMeans ###

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2
from models.mouridsen import get_AIF_KMeans
from utils.datahandling_utils import crop_image
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_MCA_prediction(model, data: pd.DataFrame, intensities: np.array, device: str, crop: float, pred_type: str):
    model.eval()
    patient = np.unique(data['Patient'])[0]
    series_idx = np.unique(data['Series'])[0]
    volume = int(np.squeeze(intensities[np.where(
        (intensities[:, 0] == patient) & (intensities[:, 1] == series_idx))[0]])[2])
    slices = data.loc[(data['Patient'] == patient) &
                      (data['Volume'] == volume)].copy()
    slice_preds = []
    info = {}
    for i, (_, slice) in enumerate(slices.iterrows()):
        slice_num = slice['Slice']
        image = pdc.dcmread(slice['ImagePath'])
        if i == 3:
            info['FA'] = image.FlipAngle
            info['TR'] = image.RepetitionTime
            info['TD'] = image.EchoTime
        image = crop_image(image.pixel_array, crop)
        image = image/np.max(image)
        image = np.expand_dims(image, axis=(0, 1))
        image = torch.tensor(image, dtype=torch.float).to(device)
        pred = torch.sigmoid(model(image)).item()
        if pred >= 0.5:
            slice_preds.append((slice_num, pred))
    if pred_type == "single":
        slice_preds = sorted(slice_preds, key=itemgetter(0), reverse=True)
        idx = len(slice_preds) // 2
        return [slice_preds[idx][0]], info
    elif pred_type == "sigmoid":
        slice_preds = sorted(slice_preds, key=itemgetter(1), reverse=True)
        return [slice_preds[0][0]], info
    return [i[0] for i in slice_preds], info

def filter_slices_on_mca_prediction(data, patient_num: int, mca_slices: list, crop=1) -> dict:
    images = {}
    for slice_idx in mca_slices:
        slice_paths = data[(data['Patient'] == patient_num)
                           & (data['Slice'] == slice_idx)]
        slice_paths = slice_paths.sort_values(
            ['Patient', 'Series', 'Volume', 'Slice'], ignore_index=True)
        image = []
        for path in slice_paths['ImagePath']:
            image.append(pdc.read_file(path).pixel_array)
        images[slice_idx] = crop_image(np.array(image), crop)
    return images

def get_AIF_label(map_path: str, aif_folder_path: str, patient: int) -> np.ndarray:
    data = pd.read_excel(map_path)
    curve_path = str(data[data['Patient'] == patient]['AIF'].item())
    if curve_path == 'None' or str(curve_path) == 'nan':
        return None    
    folders = next(os.walk(aif_folder_path))[1]
    folder_name = next(filter(lambda x: x == curve_path + '_output', folders))
    try:
        curve = np.loadtxt(aif_folder_path+'/'+folder_name+'/AIF.txt')
    except:
        curve = pd.read_csv(aif_folder_path+'/'+folder_name+'/AIF.txt', delimiter=' ', decimal=',', header=None)
        curve = curve.to_numpy().flatten()
    return curve

def extract_AIF_mouridsen(images: dict, series_idx: int, SEED: int, visualize: bool, pred_type: str, min_voxels: int) -> dict:
    curves = {}
    if pred_type == 'global':
        image = np.stack(images.values())
        curves_coords = get_AIF_KMeans(
            image, SEED, min_voxels=min_voxels, visualize=False)
        for slice_idx in images.keys():
            curves[slice_idx] = curves_coords
    else:
        for slice_idx in images.keys():
            image = images[slice_idx]
            curves_coords = get_AIF_KMeans(
                image, SEED, min_voxels=min_voxels, visualize=visualize)
            curves[slice_idx] = curves_coords
    return curves

def get_label_curves(image_data, label_path='AIF_images.xlsx', curve_path='D:\AIFs\AIFs\durable\BorrSci_MR_Data\Output'):
    train_patients = np.unique(image_data['Patient'])
    label_curves = {}
    for p in train_patients:
        c = get_AIF_label(label_path, curve_path, patient=p)
        label_curves[p] = c
    return label_curves

def visualize_curves(curves, curve_coord, patient_slice, TR, FA, signal_to_relaxation):
    if signal_to_relaxation is None:
        signal_to_relaxation = pass_through()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,8))
    fig.suptitle("Final AIF and voxel-locations")
    for c in curves:
        ax1.plot(signal_to_relaxation(c, TR, FA), alpha=0.2)
    ax1.plot(signal_to_relaxation(np.mean(curves, axis=0), TR, FA), color="red")
    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("Amplitude[a.u.]")
    ax1.grid()

    ax2.imshow(patient_slice)
    ax2.axis("off")
    for i in curve_coord:
        ax2.plot(i[1], i[0], ".r", markersize=1, linewidth = 0)

def get_info(serie):
    info = {}
    img = serie.iloc[0]['ImagePath']
    image = pdc.read_file(img)
    info['FA'] = image.FlipAngle
    info['TR'] = image.RepetitionTime
    info['TD'] = image.EchoTime
    return info

def filter_slices_on_mca_prediction(data, patient_num: int, series, mca_slices: list, crop: int):
    image_paths = get_image_volume(data, patient_num, series, mca_slices)
    image = create_image_array(image_paths, crop)
    return image

def extract_AIF_mouridsen(images, SEED: int):
    kmeans = mouridsen2006(seed=SEED)
    curves = kmeans.predict(images)
    mask = kmeans.get_mask()
    return curves, mask

def get_pred_curves(model, image_data: pd.DataFrame, patients: List, vol_intensities: np.ndarray, device: str, seed: int, crop: float, pred_type: str):
    if pred_type not in ["single", "all", "global", "sigmoid"]:
        raise Exception(
            f"Arg: one_pred must be one of \"single\", \"all\", \"global\", \"sigmoid\". Given argument: \"{pred_type}\"")
    total_curves = {}
    all_masks = {}
    slice_info_dict = {}
    for patient in patients:
        patient_curves = {}
        patient_info = {}
        patient_mask = {}
        images = image_data[image_data['Patient'] == patient]
        for s in np.unique(images['Series']):
            serie = images[images['Series'] == s]

            if pred_type == 'global':
                MCA_slices = np.unique(images['Slice'])
                slice_info = get_info(serie)
                filtered_slices = filter_slices_on_mca_prediction(
                    serie, patient, s,  MCA_slices, crop)
            else:
                MCA_slices, slice_info = get_MCA_prediction(
                    model, serie, vol_intensities, device, crop, pred_type)
                filtered_slices = filter_slices_on_mca_prediction(
                    serie, patient, s, MCA_slices, crop)

            pred_curves, pred_mask = extract_AIF_mouridsen(filtered_slices, seed)

            total_mask = np.full((pred_mask.shape[2], pred_mask.shape[3]), False)
            for i in range(pred_mask.shape[1]):
                total_mask = np.bitwise_or(total_mask, pred_mask[1, i, ...])

            patient_curves[s] = pred_curves
            patient_mask[s] = [total_mask, MCA_slices]
            patient_info[s] = slice_info
            
        total_curves[patient] = patient_curves
        all_masks[patient] = patient_mask
        slice_info_dict[patient] = patient_info

    return total_curves, slice_info_dict, all_masks    