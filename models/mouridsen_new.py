import numpy as np
import sklearn
import os

class mouridsen2006():
    def __init__(self, seed, auc_frac = 0.9, irr_frac = 0.75, select_fun = None):
        self.seed = seed
        self.auc_frac = auc_frac
        self.irr_frac = irr_frac
        self.mask = None
        self.shapes = {'result_volume' : None, 'filtered_volume' : None, 'full_volume' : None}
        self.masks = {'auc_mask' : None, 'irr_mask' : None}
        self.selected = []
        self.labels = []
        if select_fun is not None:
            self.select_fun = select_fun
        else:
            self.select_fun = self.select_curve

    def get_auc_frac(self, volume):
        auc = np.sum(volume, axis = 0)
        return np.quantile(auc, self.auc_frac), auc

    def standardize_curves_on_area(self, volume, areas):
        volume = volume / areas
        return volume

    def get_irr_frac(self, volume):
        gradient = volume[1:] - volume[:-1]
        gradient2 = gradient[1:] - gradient[:-1]
        roughness = np.sum((gradient2*gradient2)**2, axis=0)
        return np.quantile(roughness, self.irr_frac), roughness

    def filter_volume(self, arr):
        auc_frac, auc = self.get_auc_frac(arr)
        self.masks['auc_mask'] = auc > auc_frac

        volume = self.standardize_curves_on_area(arr, auc)
        self.shapes['full_volume'] = volume.shape

        _, rough = self.get_irr_frac(volume)
        roughness_frac, _ = self.get_irr_frac(volume[:, self.masks['auc_mask']])
        self.masks['irr_mask'] = rough < roughness_frac

        filter_volume = volume[:, self.masks['auc_mask'] & self.masks['irr_mask']]
        self.shapes['filtered_volume'] = filter_volume.shape
        filter_volume = np.swapaxes(filter_volume, 0, 1)

        return filter_volume
    
    def select_curve(self, curves):
        # Selects curve with highest peak/AUC
        curve_peak = np.max(curves, axis=1)
        AUC = np.sum(curves, axis=1)
        idx = np.argmax(curve_peak / AUC)
        return idx
    
    def perform_kmeans(self, arr):
        os.environ["OMP_NUM_THREADS"] = '1'
        (_centroid, label, _inertia) = sklearn.cluster.k_means(arr, 5, n_init=10, random_state=self.seed)
        curves = np.zeros((5, arr.shape[1]))
        for i in range(5):
            curves[i] = np.average(arr[label==i], axis = 0)
        select_idx = self.select_fun(curves)
        next_volume = arr[label==select_idx]
        self.selected.append(select_idx)
        self.labels.append(label)
        return next_volume
    
    def get_mask(self):
        mask = np.full((self.shapes['result_volume']), False)
        mask[:, self.labels[1] == self.selected[1]] = True
        mask1 = np.full(self.shapes['filtered_volume'], False)
        mask1[:, self.labels[0] == self.selected[0]] = mask
        mask2 = np.full(self.shapes['full_volume'], False)
        mask2[:, self.masks['auc_mask'] & self.masks['irr_mask']] = mask1
        self.mask = mask2
        return mask2

    def predict(self, array):
        filtered_volume = self.filter_volume(array)
        first_kmeans = self.perform_kmeans(filtered_volume)
        self.shapes['result_volume'] = (first_kmeans.shape[1], first_kmeans.shape[0])
        last_kmeans = self.perform_kmeans(first_kmeans)
        return last_kmeans
    
