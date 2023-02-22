import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from typing import List

def filter_AUC(slices:np.ndarray, frac = 0.9):
    areas = np.zeros((slices.shape[1], slices.shape[2]))
    for row in range(slices.shape[1]):
        for col in range(slices.shape[2]):
            area = np.trapz(slices[:, row, col], dx=1)
            areas[row, col] = area
    areas_flat = np.sort(areas, axis=None)[::-1]
    areas_flat = areas_flat[:int(slices.shape[1]*slices.shape[2]*(1-frac))]
    auc_filter = areas > np.min(areas_flat)
    return auc_filter, areas

def standardize_curves_on_area(slices:np.ndarray, areas:np.ndarray):
    areas_stacked = np.dstack([areas]*slices.shape[0])
    areas_stacked = np.transpose(areas_stacked, (2, 0, 1))
    standardized_curves = slices/areas_stacked
    return standardized_curves

def get_curve_map(slices:np.ndarray):
    locations_dict = {}
    for row in range(slices.shape[1]):
        for col in range(slices.shape[2]):
        # Arrays needs to be hashed manually
            locations_dict[hash(slices[:, row, col].tobytes())] = (row, col)
    return locations_dict

def roughness(X):
    gradient_2 = np.gradient(np.gradient(X))
    area = np.trapz(gradient_2**2, dx=1)
    return area

def filter_std_curves_roughness(slices:np.ndarray, auc_filter:np.ndarray, frac = 0.25):
    curve_roughness = np.zeros((slices.shape[1], slices.shape[2]))
    for row in range(slices.shape[1]):
        for col in range(slices.shape[2]):
            ruffle = roughness(slices[:, row, col])
            curve_roughness[row, col] = ruffle
    roughness_flat = curve_roughness[auc_filter].flatten()
    roughness_flat = np.sort(roughness_flat)
    roughness_flat = roughness_flat[:int(len(roughness_flat)*(1-frac))]
    roughness_filter = curve_roughness < np.max(roughness_flat)
    roughness_filter = np.bitwise_and(roughness_filter, auc_filter)
    return roughness_filter

def select_curves_on_filter(slices:np.ndarray, selection_filter:np.ndarray):
    next_curves = []
    for row in range(slices.shape[1]):
        for col in range(slices.shape[2]):
            if selection_filter[row, col]:
                next_curves.append(slices[:, row, col])
    return next_curves

def kMeans_on_curves(next_curves:List[np.ndarray], seed:int, visualize:bool, clusters = 5, max_voxels = 20):
    os.environ["OMP_NUM_THREADS"] = '1'
    while len(next_curves) > max_voxels:
        kMeans_clustering = KMeans(n_clusters = clusters, random_state=seed, n_init=10).fit(next_curves)
        cluster_averaged_curves = np.zeros((clusters, 80))
        mapped_curves = {n: [] for n in range(clusters)}
        for i, curve in zip(kMeans_clustering.labels_, next_curves):
            cluster_averaged_curves[i] += curve
            mapped_curves[i].append(curve)
        _, num_in_clusters = np.unique(kMeans_clustering.labels_, return_counts = True)
        for i in range(clusters):
            cluster_averaged_curves[i] /= num_in_clusters[i] 
        # first_moment = np.trapz(cluster_averaged_curves*(np.arange(0, 80, 1)**2), axis = 1)
        # lowest_idx = np.argmin(first_moment)

        # Select curve with the highest peak
        curve_max = np.argmax(cluster_averaged_curves, axis = 1)
        value_max = np.zeros(curve_max.shape)
        for i in range(0, cluster_averaged_curves.shape[0]):
            value_max[i] = np.max(cluster_averaged_curves[i, :])
        lowest_idx = np.argmax(value_max)

        next_curves = mapped_curves[lowest_idx]
        if visualize:
            fig = plt.figure()
            for i in range(clusters):
                # cluster_averaged_curves[i] /= num_in_clusters[i]
                plt.plot(cluster_averaged_curves[i], label=f"Cluster {i}")
                plt.title(f'Selected cluster: {lowest_idx}')
            plt.legend()
    return next_curves

def plot_filtered_curves(slices:np.ndarray, filter:np.ndarray):
    fig = plt.figure()
    for row in range(slices.shape[1]):
        for col in range(slices.shape[2]):
            if filter[row, col]:
                plt.plot(slices[:, row, col])

def get_AIF_KMeans(slices:np.ndarray, seed: int, max_voxels: int = 20, visualize: bool = False):
    # Get S_0
    # Filter the curves with the largest AUC
    auc_filter, areas = filter_AUC(slices)
    # Standardize the curves so all AUCs are equal to 1
    standardized_curves = standardize_curves_on_area(slices, areas)
    # Map curves to their coordinate to view where the final AIF is extracted from
    curve_map = get_curve_map(standardized_curves)
    #TODO: The roughness filter has not been implemented.
    # Filter out 25% of the most irregular curves
    # rough_auc_filter = filter_std_curves_roughness(standardized_curves, auc_filter)
    # Get the standardized curves with the highest AUCs
    filtered_curves = select_curves_on_filter(standardized_curves, auc_filter)
    # Perform KMeans to find the best curve
    selected_curves = kMeans_on_curves(filtered_curves, seed, visualize, max_voxels=max_voxels)
 
    coordinates = []
    final_curves = []
    for curve in selected_curves:
        # Hashing needs to be done manually
        coordinate = curve_map[hash(curve.tobytes())]
        coordinates.append(coordinate)
        final_curves.append(slices[:, coordinate[0], coordinate[1]])

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,7))
        fig.suptitle("Final AIF and voxel-locations")
        ax1.plot(np.mean(final_curves, axis=0))
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Amplitude(C(t))")
        ax1.grid()

        ax2.imshow(slices[0, :, :])
        ax2.axis("off")
        for i in coordinates:
            ax2.plot(i[1], i[0], ".r")

    return [final_curves, coordinates]

if __name__ == "__main__":
    import pandas as pd
    from utils.preprocess_utils import get_slice_volumes_for_one_patient
    import pydicom as pdc
    SEED = 42

    data = pd.read_csv("data_indices.csv", ",", index_col = 0)
    data = data.sort_values(["Patient", "Volume", "Slice"], ignore_index=True)

    slice_paths = get_slice_volumes_for_one_patient(data, 1, 20)

    slices = []
    for path in slice_paths['ImagePath']:
        image = pdc.read_file(path)
        slices.append(image.pixel_array)
    slices = np.array(slices)

    _ = get_AIF_KMeans(slices, SEED, visualize=True)