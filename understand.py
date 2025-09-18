import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import io
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = 'MobiFall_Dataset'
TARGET_SAMPLING_RATE_HZ = 50
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms"
SEQUENCE_LENGTH = int(TARGET_SAMPLING_RATE_HZ * 4)
STEP = int(TARGET_SAMPLING_RATE_HZ * 1)

SENSOR_CODES = ["acc", "gyro", "ori"]
EXPECTED_COLUMNS = {
    "acc" : ["acc_x", "acc_y", "acc_z"],
    "gyro" : ["gyro_x", "gyro_y", "gyro_z"],
    "ori": ["ori_azimuth", "ori_pitch", "ori_roll"]
}

ALL_FEATURE_COLUMNS = [
    "acc_x", "acc_y", "acc_z", "acc_smv",
    "gyro_x", "gyro_y", "gyro_z", "gyro_smv",
    "ori_azimuth", "ori_pitch", "ori_roll"
]



def load_and_resample_sensor_file(filepath, sensor_code):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data_start_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().upper() == "@DATA":
                data_start_line_index = i + 1
                break
        if data_start_line_index == -1 or data_start_line_index >= len(lines):
            return None
        
        data_string = "".join(lines[data_start_line_index:])
        if not data_string.strip():
            return None
        
        df = pd.read_csv(io.StringIO(data_string), header=None, usecols=[0, 1, 2, 3])
        if df.empty:
            return None
        
        df.columns = ['timestamp_ns'] + EXPECTED_COLUMNS[sensor_code]
        df['timestamp'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
        df = df.set_index('timestamp').drop(columns=['timestamp_ns'])
        df = df.sort_index()

        df_resampled = df.resample(TARGET_SAMPLING_PERIOD).mean().interpolate(method='linear', limit_direction='both')

        if sensor_code == 'acc':
            if all(col in df_resampled.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                df_resampled['acc_smv'] = np.sqrt(df_resampled['acc_x']**2 + df_resampled['acc_y']**2 + df_resampled['acc_z']**2)
        elif sensor_code == 'gyro':
            if all(col in df_resampled.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                df_resampled['gyro_smv'] = np.sqrt(df_resampled['gyro_x']**2 + df_resampled['gyro_y']**2 + df_resampled['gyro_z']**2)

        
        return df_resampled
    
    except (pd.errors.EmptyDataError, ValueError):
        return None
    except Exception as e:
        print(f"Error processing file {filepath}: {e}. Skipping")
        return None
    

def load_data_from_structured_folders(dataset_root_path):
    print(f"Scanning for data in: {dataset_root_path}")
    if not os.path.isdir(dataset_root_path):
        print(f"ERROR: Dataset root path '{dataset_root_path}' not found.")
        return [], []
    
    trial_sensor_files_map = defaultdict(lambda: defaultdict(str))
    trial_metadata_map = {}

    for dirpath, _, filenames in os.walk(dataset_root_path):
        relative_path = os.path.relpath(dirpath, dataset_root_path)
        path_parts = relative_path.split(os.sep)
        if len(path_parts) != 4:continue

        for filename in filenames:
            if not filename.endswith(".txt"): continue
            
            fname_parts = filename.replace('.txt', '').split('_')
            if len(fname_parts) != 4: continue

            _, sensor_code, _, trial_no_str = fname_parts
            sensor_code = sensor_code.lower()
            if sensor_code not in SENSOR_CODES: continue

        

        