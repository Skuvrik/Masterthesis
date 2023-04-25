import torch
from torch import nn
import numpy as np
import pydicom as pdc
from typing import Optional, Callable
from utils.datahandling_utils import crop_image

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
        image = np.expand_dims(image, axis=(0,1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.to(device)
        actual = labels[(labels['Patient'] == int(selected['Patient'])) & (labels['Slice'] == int(selected['Slice']))]['Label'].item()
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

def NLC_s2r(S, TR, FA):
    FA = FA*(np.pi/180) # degree -> rad
    TR = TR / 1000 # ms -> s
    T_10 = 1.4
    S_0 = np.mean(S[:5])
    E_10 = np.exp(-TR/T_10)
    B = (1-E_10)/(1-E_10*np.cos(FA))
    A = np.nan_to_num(B*S/S_0)
    R1 = np.nan_to_num(-1/TR * np.log((1-A)/(1-A*np.cos(FA))))
    return R1 - np.mean(R1[:5])

def get_MCA_prediction(model, data: pd.DataFrame, intensities: np.array, device: str, crop: float, one_pred: bool = True):
    model.eval()
    patient = np.unique(data['Patient'])[0]
    volume = int(np.squeeze(intensities[np.where(intensities[:, 0] == patient)[0]])[1])
    slices = data.loc[(data['Patient']==patient) & (data['Volume']==volume)].copy()
    slice_preds = []
    info = {}
    for i, (_, slice) in enumerate(slices.iterrows()):
        slice_num = slice['Slice']
        image = pdc.dcmread(slice['ImagePath'])
        if i == 3:
            info['FA'] = image.FlipAngle
            info['TR'] = image.RepetitionTime
        image = crop_image(image.pixel_array, crop)
        image = image/np.max(image)
        image = np.expand_dims(image, axis=(0,1))
        image = torch.tensor(image, dtype=torch.float).to(device)
        pred = int(round(torch.sigmoid(model(image)).item()))
        if pred == 1: slice_preds.append(slice_num)
    if len(slice_preds) > 1 and one_pred:
        slice_preds = sorted(slice_preds, reverse=True)
        idx = len(slice_preds) // 2
        return [slice_preds[idx]], info
    return slice_preds, info

def filter_slices_on_mca_prediction(data, patient_num: int, mca_slices: list) -> dict:
    images = {}
    for slice_idx in mca_slices:
        slice_paths = data[(data['Patient']==patient_num)&(data['Slice']==slice_idx)]
        slice_paths = slice_paths.sort_values(['Patient', 'Volume', 'Slice'], ignore_index=True)
        image = []
        for path in slice_paths['ImagePath']:
            image.append(pdc.read_file(path).pixel_array)
        images[slice_idx]=np.array(image)
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

def extract_AIF_mouridsen(images: dict, SEED: int, info: dict, visualize:bool = False) -> dict:
    curves = {}
    for slice_idx in images.keys():
        image = images[slice_idx]
        curves_coords = get_AIF_KMeans(image, SEED, max_voxels=20, visualize=visualize)
        curves[slice_idx] = curves_coords
    return curves

def get_label_curves(image_data, label_path='AIF_images.xlsx', curve_path='D:\AIFs\AIFs\durable\BorrSci_MR_Data\Output'):
    train_patients = np.unique(image_data['Patient'])
    label_curves = {}
    for p in train_patients:
        c = get_AIF_label(label_path, curve_path, patient=p)
        label_curves[p] = c
    return label_curves

def pass_through(x):
    return x

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

def single_eval_curve(model, image_data:pd.DataFrame, patient:int, vol_intensities:np.ndarray, device: str, seed: int, crop: float, visualize: bool, one_pred: bool):
    images = image_data[image_data['Patient'] == patient]
    MCA_slices, slice_info = get_MCA_prediction(model, images, vol_intensities, device, crop, one_pred)
    filtered_slices = filter_slices_on_mca_prediction(images, patient, MCA_slices)
    pred_curves = extract_AIF_mouridsen(filtered_slices, seed, slice_info, visualize)
    return pred_curves, slice_info        
    
def eval_curves(model, image_data:pd.DataFrame, vol_intensities:np.ndarray, device: str, seed: int, crop: float, visualize: bool, signal_to_relaxation: Callable = None, one_pred: bool = True):
    actual_curves = get_label_curves(image_data)
    results = []
    extractions = {}
    labels = {}
    for p in actual_curves.keys():
        print(f"Patient {p}")
        pred_curves, slice_info = single_eval_curve(model, image_data, p, vol_intensities, device, seed, crop, visualize, one_pred)
        for s in pred_curves.keys():
            if actual_curves[p] is None:
                results.append([p, s, None, None, None])
                continue

            if visualize:
                volume = int(np.squeeze(vol_intensities[np.where(vol_intensities[:, 0] == p)[0]])[1])
                slice = image_data[((image_data['Patient'] == p) & (image_data["Slice"] == s)) & (image_data['Volume'] == volume)]
                image = pdc.dcmread(slice['ImagePath'].item()).pixel_array
                visualize_curves(pred_curves[s][0], pred_curves[s][1], image, slice_info['TR'], slice_info['FA'], signal_to_relaxation)

            pred = np.mean(pred_curves[s][0], axis = 0)
            if signal_to_relaxation is not None:
                pred = signal_to_relaxation(pred, slice_info['TR'], slice_info['FA'])
            extractions[(p,s)] = pred
            labels[(p,s)] = actual_curves[p]

            results.append([p, s, mae(actual_curves[p], pred), mse(actual_curves[p], pred, squared=False), r2(actual_curves[p], pred)])
    return pd.DataFrame(results, columns=['Patient', 'Slice', 'MAE', 'RMSE', 'R2'], index=None), extractions, labels