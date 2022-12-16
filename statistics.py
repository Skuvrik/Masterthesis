import os
import numpy as np
import pydicom as pdc
import pandas as pd
from pydicom.fileset import FileSet
import matplotlib.pyplot as plt

def get_image_stats():

    dcm_statistics = pd.DataFrame(columns=["PatientID",
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
                                        "Rows",
                                        "Columns"])

    for root, dirs, files in os.walk("D:/iCAT_IMAGES/"):
        if(len(dirs) < 1):
            path = '/'
            path_list = (root+'/'+files[0]).split('\\')
            path = path.join(path_list)
            ds = pdc.dcmread(path)
            dcm_statistics.loc[len(dcm_statistics.index)] = [ds.PatientID,
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
                                                            ds.Rows,
                                                            ds.Columns]
    return dcm_statistics

def get_patient_stats():

    dcm_statistics = pd.DataFrame(columns=["PatientID",
                                        "PatientAge", 
                                        "PatientSex", 
                                        "PatientWeight", 
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
                                        "Rows",
                                        "Columns"])

    for root, dirs, files in os.walk("D:/iCAT_IMAGES/"):
        if(len(dirs) < 1):
            path = '/'
            path_list = (root+'/'+files[0]).split('\\')
            path = path.join(path_list)
            ds = pdc.dcmread(path)
            dcm_statistics.loc[len(dcm_statistics.index)] = [ds.PatientID,
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
                                                            ds.Rows,
                                                            ds.Columns]
    return dcm_statistics