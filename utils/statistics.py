import os
import numpy as np
import pydicom as pdc
import pandas as pd
from pydicom.fileset import FileSet
import matplotlib.pyplot as plt
from typing import List

def get_image_stats(data, exclusions=None):
    dcm_statistics = pd.DataFrame(columns=["Patient",
                                        "PatientID",
                                        "AcquisitionDate", 
                                        "Manufacturer", 
                                        "MRAcquisitionType", 
                                        "RepetitionTime",
                                        "EchoTime",
                                        "FlipAngle",
                                        "ScanningSequence",
                                        "Modality",
                                        "SequenceName",
                                        "SequenceVariant",
                                        "SliceThickness",
                                        "ContrastBolusAgent",
                                        "ContrastBolusVolume",
                                        "Manufacturer",
                                        "MAnufacturerModelName"])
    for patient in np.unique(data['Patient']):
        if exclusions != None:
            if patient in exclusions: continue
        path = data[data['Patient'] == patient]['ImagePath'].iloc[0]
        ds = pdc.dcmread(path)
        dcm_statistics.loc[len(dcm_statistics.index)] = [patient,
                                                        ds.PatientID,
                                                        ds.AcquisitionDate,
                                                        ds.Manufacturer,
                                                        ds.MRAcquisitionType,
                                                        ds.RepetitionTime,
                                                        ds.EchoTime,
                                                        ds.FlipAngle,
                                                        ds.ScanningSequence,
                                                        ds.Modality,
                                                        ds.SequenceName,
                                                        ds.SequenceVariant,
                                                        ds.SliceThickness,
                                                        ds.ContrastBolusAgent,
                                                        ds.ContrastBolusVolume,
                                                        ds.Manufacturer,
                                                        ds.ManufacturerModelName]
    return dcm_statistics

def get_patient_stats(data, exclusions=None):

    dcm_statistics = pd.DataFrame(columns=["PatientID",
                                        "PatientAge", 
                                        "PatientSex", 
                                        "PatientWeight", 
                 ])

    for patient in np.unique(data['Patient']):
        if exclusions != None:
            if patient in exclusions: continue
        path = data[data['Patient'] == patient]['ImagePath'].iloc[0]
        ds = pdc.dcmread(path)
        dcm_statistics.loc[len(dcm_statistics.index)] = [ds.PatientID,
                                                            ds.PatientAge,
                                                            ds.PatientSex,
                                                            ds.PatientWeight,
                                                            ]
    return dcm_statistics

def bland_altman(predicted, actual):
    predicted = np.array(list(predicted.values()))
    actual = np.array(list(actual.values()))
    diff = predicted - actual
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff
    print(f"Limits of agreement upper: {loa_upper:.2f} lower: {loa_lower:.2f}")

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(np.mean([predicted,actual], axis=0), diff, s=1)
    plt.axhline(mean_diff, color='gray', linestyle='--')
    plt.axhline(loa_lower, color='gray', linestyle='--')
    plt.axhline(loa_upper, color='gray', linestyle='--')
    plt.xlabel('Mean of given and extracted AIFs')
    plt.ylabel('Difference between extracted and given AIFs')
    plt.title('Bland-Altman Plot')

    # Show the plot
    plt.show()

def get_ROC_curve():
    pass
