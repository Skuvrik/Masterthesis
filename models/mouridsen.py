import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from typing import List
import operator

def filter_AUC(volume: np.ndarray, frac=0.9):
    areas = np.zeros((volume.shape[1], volume.shape[2]))
    for row in range(volume.shape[1]):
        for col in range(volume.shape[2]):
            area = np.trapz(volume[:, row, col], dx=1)
            areas[row, col] = area
    areas_flat = np.sort(areas, axis=None)[::-1]
    areas_flat = areas_flat[:int(volume.shape[1]*volume.shape[2]*(1-frac))]
    auc_filter = areas > np.min(areas_flat)
    return auc_filter, areas

def standardize_curves_on_area(volume: np.ndarray, areas: np.ndarray):
    areas_stacked = np.dstack([areas]*volume.shape[0])
    areas_stacked = np.transpose(areas_stacked, (2, 0, 1))
    standardized_curves = np.nan_to_num(volume/areas_stacked)
    return standardized_curves

def get_curve_map(volume: np.ndarray):
    locations_dict = {}
    for row in range(volume.shape[1]):
        for col in range(volume.shape[2]):
            # Arrays needs to be hashed manually
            locations_dict[hash(volume[:, row, col].tobytes())] = (row, col)
    return locations_dict

def roughness(X):
    gradient = X[1:] - X[:-1]
    gradient2 = gradient[1:] - gradient[:-1]
    roughness = np.sum((gradient2*gradient2)**2)
    return roughness

def filter_std_curves_roughness(slices: np.ndarray, auc_filter: np.ndarray, frac=0.25):
    curve_roughness = np.zeros((slices.shape[1], slices.shape[2]))
    for row in range(slices.shape[1]):
        for col in range(slices.shape[2]):
            curve_roughness[row, col] = roughness(slices[:, row, col])
    roughness_flat = curve_roughness[auc_filter].flatten()
    roughness_flat = np.sort(roughness_flat)
    roughness_flat = roughness_flat[:int(len(roughness_flat)*(1-frac))]
    roughness_filter = curve_roughness < np.max(roughness_flat)
    roughness_filter = np.bitwise_and(roughness_filter, auc_filter)
    return roughness_filter

def select_curves_on_filter(volume: np.ndarray, selection_filter: np.ndarray):
    next_curves = []
    for row in range(volume.shape[1]):
        for col in range(volume.shape[2]):
            if selection_filter[row, col]:
                next_curves.append(volume[:, row, col])
    return next_curves

def get_FWHM(x: np.ndarray, y: np.ndarray, height: float = 0.5) -> float:
    height_half_max = np.max(y) * height
    index_max = np.argmax(y)
    x_low = np.interp(height_half_max, y[:index_max+1], x[:index_max+1])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))

    return x_high - x_low

def FWHM_TTP_PV_select(curves):
    FWHM = np.zeros(curves.shape[0])
    PV = np.max(curves, axis=1)
    TTP = np.argmax(curves, axis=1)
    for i, c in enumerate(curves):
        FWHM[i] = get_FWHM(list(range(0,len(c))), c)
    idx = np.argmax(PV / TTP*FWHM)
    return idx

def select_first_and_highest_peak(curves: List[np.ndarray]) -> int:
    highest_peak = np.max(curves)
    peak_tresh = highest_peak * 0.7
    peak_map = {}
    index_peak = []
    for i in range(0, curves.shape[0]):
        index = np.argmax(curves[i, :])
        peak = np.max(curves[i, :])
        if peak >= peak_tresh:
            peak_map[(index, peak)] = i
            index_peak.append([index, peak])
    sorted_index_peak = sorted(sorted(index_peak, key=operator.itemgetter(
        1), reverse=True), key=operator.itemgetter(0))
    idx = peak_map[(sorted_index_peak[0][0], sorted_index_peak[0][1])]
    return idx

def select_curve_lowest_moment(curves: List[np.ndarray]) -> int:
    """
    Finds the curve with the lowest first moment:
    first_moment = integral_0^(-1) (x-TTP)*f(x) dx
    """
    curve_TTP = np.expand_dims(np.argmax(curves, axis=1), 1)
    x_range = np.tile(
        np.arange(start=0, stop=curves.shape[1]), (curves.shape[0], 1))
    x_minus_ttp = x_range - curve_TTP
    first_moment = np.sum(curves*x_minus_ttp, axis=1)
    idx = np.argmin(first_moment)
    return idx

def select_height_divided_by_AUC(curves: List[np.ndarray]) -> int:
    curve_peak = np.max(curves, axis=1)
    AUC = np.sum(curves, axis=1)
    idx = np.argmax(curve_peak / AUC)
    return idx

def plot_filtered_curves(volume: np.ndarray, filter: np.ndarray):
    fig = plt.figure()
    for row in range(volume.shape[1]):
        for col in range(volume.shape[2]):
            if filter[row, col]:
                plt.plot(volume[:, row, col])

def kMeans_on_curves(next_curves: List[np.ndarray], seed: int, visualize: bool, min_voxels, clusters=5):
    os.environ["OMP_NUM_THREADS"] = '1'
    length = len(next_curves[0])
    for _ in range(2):
        kMeans_clustering = KMeans(
            n_clusters=clusters, random_state=seed, n_init=10).fit(next_curves)
        cluster_averaged_curves = np.zeros((clusters, length))
        mapped_curves = {n: [] for n in range(clusters)}
        for i, curve in zip(kMeans_clustering.labels_, next_curves):
            cluster_averaged_curves[i] += curve
            mapped_curves[i].append(curve)
        _, num_in_clusters = np.unique(
            kMeans_clustering.labels_, return_counts=True)
        for i in range(clusters):
            cluster_averaged_curves[i] /= num_in_clusters[i]

        # Select curve
        idx = select_height_divided_by_AUC(cluster_averaged_curves)
        if len(mapped_curves[idx]) < min_voxels:
            break
        next_curves = mapped_curves[idx]

        if visualize:
            fig = plt.figure()
            for i in range(clusters):
                plt.plot(cluster_averaged_curves[i], label=f"Cluster {i}")
                plt.title(f'Selected cluster: {idx}')
            plt.legend()
    return next_curves

def _get_AIF_KMeans_single(volume, seed, visualize, min_voxels):
    # Filter the curves with the largest AUC
    auc_filter, areas = filter_AUC(volume)
    # Standardize the curves so all AUCs are equal to 1
    standardized_curves = standardize_curves_on_area(volume, areas)
    # Map curves to their coordinate to view where the final AIF is extracted from
    curve_map = get_curve_map(standardized_curves)
    # Filter out 25% of the most irregular curves
    rough_auc_filter = filter_std_curves_roughness(standardized_curves, auc_filter)
    # Get the standardized curves with the highest AUCs
    filtered_curves = select_curves_on_filter(standardized_curves, rough_auc_filter)
    # Perform KMeans to find the best curve
    return kMeans_on_curves(filtered_curves, seed, visualize, min_voxels), curve_map

def _get_AIF_KMeans_global(volumes, seed, visualize, min_voxels):
    curves = []
    pred_maps = []
    for volume_idx in range(volumes.shape[0]):
        pred_curves, curve_map = _get_AIF_KMeans_single(
            volumes[volume_idx], seed, visualize, min_voxels)
        curves.extend(pred_curves)
        pred_maps.append(curve_map)
    final_curves = kMeans_on_curves(curves, seed, visualize, min_voxels)
    return final_curves, pred_maps

def get_AIF_KMeans(volumes: np.ndarray, seed: int, min_voxels: int = 10, visualize: bool = False):
    coordinates = []
    final_curves = []

    if len(volumes.shape) == 4:
        selected_curves, curve_maps = _get_AIF_KMeans_global(
            volumes, seed, visualize, min_voxels)
        for curve in selected_curves:
            # Hashing needs to be done manually
            for v, c_map in enumerate(curve_maps):
                if hash(curve.tobytes()) in c_map.keys():
                    coordinate = c_map[hash(curve.tobytes())]
                    coordinates.append(coordinate)
                    final_curves.append(
                        volumes[v, :, coordinate[0], coordinate[1]])
    else:
        selected_curves, curve_maps = _get_AIF_KMeans_single(
            volumes, seed, visualize, min_voxels)
        for curve in selected_curves:
            # Hashing needs to be done manually
            coordinate = curve_maps[hash(curve.tobytes())]
            coordinates.append(coordinate)
            final_curves.append(volumes[:, coordinate[0], coordinate[1]])

    return [final_curves, coordinates]