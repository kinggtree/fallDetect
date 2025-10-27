import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --- Configuration ---
DATASET_PATH = 'MobiFall_Dataset'
TARGET_SAMPLING_RATE_HZ = 50.0  # Target sampling rate in Hz
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms"
WINDOW_SECONDS = 4
WINDOW_SIZE = int(TARGET_SAMPLING_RATE_HZ * WINDOW_SECONDS) # x samples for y seconds at 50Hz

STEP_SECONDS = 2 # x秒步长
STEP = int(TARGET_SAMPLING_RATE_HZ * STEP_SECONDS)          # 50*x samples for x second step at 50Hz

SENSOR_CODES = ["acc", "gyro", "ori"]
EXPECTED_COLUMNS = {
    "acc": ["acc_x", "acc_y", "acc_z"],
    "gyro": ["gyro_x", "gyro_y", "gyro_z"],
    "ori": ["ori_azimuth", "ori_pitch", "ori_roll"]
}
ALL_FEATURE_COLUMNS = [
    "acc_x", "acc_y", "acc_z", "acc_smv",
    "gyro_x", "gyro_y", "gyro_z", "gyro_smv",
    "ori_azimuth", "ori_pitch", "ori_roll"
]


def load_and_resample_sensor_file(filepath, sensor_code):
    """加载单个传感器文件，转换时间戳并进行重采样。"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 初始化一个变量作为“标记未找到”的标志
        data_start_line_index = -1

        # 遍历文件中的每一行
        for i, line in enumerate(lines):
            # 检查当前行是否是"@DATA"标记
            if line.strip().upper() == "@DATA":
                # 如果是，则记录下一行的行号并跳出循环
                data_start_line_index = i + 1
                break

        # 检查标记是否被找到
        if data_start_line_index == -1 or data_start_line_index >= len(lines):
            return None

        # 将数据行拼接成单个字符串
        data_string = "".join(lines[data_start_line_index:])

        # 检查字符串是否为空
        if not data_string.strip():
            return None

        # 使用pandas处理数据
        df = pd.read_csv(io.StringIO(data_string), header=None, usecols=[0, 1, 2, 3])
        
        # 检查生成的数据表是否为空
        if df.empty:
            return None

        # 为数据列进行命名
        df.columns = ['timestamp_ns'] + EXPECTED_COLUMNS[sensor_code]

        # 将ns时间戳转换为标准的日期时间格式
        df['timestamp'] = pd.to_datetime(df['timestamp_ns'], unit='ns')

        # 将新的日期时间设置为索引，并删除旧的时间戳列
        df = df.set_index('timestamp').drop(columns=['timestamp_ns'])

        # 按时间索引进行排序
        df = df.sort_index()

        # 将采样时间不均匀的传感器数据，强制转换为频率统一（每20毫秒一个点）的规整数据流，并填补其中的所有空白
        df_resampled = df.resample(TARGET_SAMPLING_PERIOD).mean().interpolate(method='linear', limit_direction='both')

        # 检查当前处理的传感器是否为加速度计 ('acc')
        if sensor_code == 'acc':
            # 安全性检查 - 确认三轴数据都存在
            if all(col in df_resampled.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                # 计算信号幅值向量 (SMV)
                df_resampled['acc_smv'] = np.sqrt(
                    df_resampled['acc_x']**2 + df_resampled['acc_y']**2 + df_resampled['acc_z']**2
                )

        # 如果不是加速度计，则检查是否为陀螺仪 ('gyro')
        elif sensor_code == 'gyro':
            # 对陀螺仪数据执行相同的操作
            if all(col in df_resampled.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                df_resampled['gyro_smv'] = np.sqrt(
                    df_resampled['gyro_x']**2 + df_resampled['gyro_y']**2 + df_resampled['gyro_z']**2
                )

        return df_resampled

    except (pd.errors.EmptyDataError, ValueError):
        return None
    except Exception as e:
        print(f"Error processing file {filepath}: {e}. Skipping.")
        return None


def load_data_from_structured_folders(dataset_root_path):
    """
    遍历数据集文件夹，处理、对齐并组合每个试验的传感器数据。
    返回一个按 subject_id 组织的数据字典。
    """
    print(f"Scanning for data in: {dataset_root_path}")
    if not os.path.isdir(dataset_root_path):
        print(f"ERROR: Dataset root path '{dataset_root_path}' not found.")
        return {} # 返回一个空字典

    # 存放每一次活动试验（trial）所对应的各个传感器文件的路径（数据文件的位置）
    trial_sensor_files_map = defaultdict(lambda: defaultdict(str)) 

    # 存放每一次活动试验的元数据（这些数据代表什么，即标签信息）
    trial_metadata_map = {} 
    
    # 遍历数据集的每一个文件夹
    for dirpath, _, filenames in os.walk(dataset_root_path): 
        # 解析文件夹路径，以确定活动类别和具体活动
        relative_path = os.path.relpath(dirpath, dataset_root_path) 
        path_parts = relative_path.split(os.sep) 
        # 确保只处理包含实际数据文件的特定层级文件夹
        if len(path_parts) != 3: continue 

        # 遍历这些特定文件夹中的每一个文件
        for filename in filenames: 
            # 确保只处理.txt文件
            if not filename.endswith(".txt"): continue 
            
            # 解析文件名，通过下划线分割以获取各个部分
            fname_parts = filename.replace('.txt', '').split('_') 
            # 过滤掉不符合预期格式的文件名
            if len(fname_parts) != 4: continue 
            
            # 从文件名部分中提取所需信息
            _, sensor_code, _, trial_no_str = fname_parts 
            # 将传感器代码转为小写以保持一致性
            sensor_code = sensor_code.lower() 
            # 确保是已知的传感器类型 ('acc', 'gyro', 'ori')
            if sensor_code not in SENSOR_CODES: continue 

            # 尝试从路径和文件名中提取并转换所有元数据
            try:
                # 从文件夹路径的第一部分提取受试者ID
                subject_match = re.fullmatch(r'sub(\d+)', path_parts[0], re.IGNORECASE) 
                if not subject_match: continue 
                subject_id = int(subject_match.group(1)) 
                
                # 从文件夹路径的第二和第三部分获取类别和活动代码
                category = path_parts[1].upper() 
                activity_code = path_parts[2].upper() 
                # 将试验编号从字符串转换为整数
                trial_no = int(trial_no_str) 
                # 构建完整的文件路径
                filepath = os.path.join(dirpath, filename)
                
                # 创建一个唯一的键来标识这次试验 (受试者, 活动, 试验编号)
                trial_key = (subject_id, activity_code, trial_no) 
                # 在映射表中存储该传感器文件的路径
                trial_sensor_files_map[trial_key][sensor_code] = filepath 
                # 如果是第一次遇到这个试验，则记录其元数据（类别和活动代码）
                if trial_key not in trial_metadata_map: 
                    trial_metadata_map[trial_key] = {"category": category, "activity_code": activity_code} 
            except (AttributeError, ValueError):
                # 如果在提取或转换过程中出现任何错误，则跳过该文件
                continue

    # *** 这是关键改动 ***
    # 初始化一个字典，用于按受试者ID存放处理好的 (数据, 标签) 元组
    subject_data_map = defaultdict(list)
    
    print(f"\nProcessing and combining {len(trial_sensor_files_map)} unique trials...")
    
    # 遍历前面组织好的每一次活动试验（trial）
    for trial_key, sensor_files in trial_sensor_files_map.items(): 
        # 确保该次试验包含了 acc, gyro, ori 全部三种传感器文件，否则跳过
        if not all(s_code in sensor_files for s_code in SENSOR_CODES): continue 

        # 使用字典推导式，为每种传感器加载并重采样数据
        resampled_dfs = {s_code: load_and_resample_sensor_file(sensor_files[s_code], s_code) for s_code in SENSOR_CODES} 
    
        # 如果任何一个文件加载或处理失败（返回了None或空表），则跳过这次试验
        if any(df is None or df.empty for df in resampled_dfs.values()): continue 

        try:
            # --- 时间对齐关键步骤 ---
            # 找到三个传感器数据中最晚的开始时间
            common_start = max(df.index.min() for df in resampled_dfs.values()) 
            # 找到三个传感器数据中最早的结束时间
            common_end = min(df.index.max() for df in resampled_dfs.values())
            # 如果没有重叠的时间窗口，则跳过
            if common_start >= common_end: continue 

            # 将三个数据表都裁剪到共同的时间范围内
            aligned_dfs = [resampled_dfs[s_code][common_start:common_end].reset_index(drop=True) for s_code in SENSOR_CODES] 
            # 确保对齐后的数据表长度一致且不为空，否则跳过
            if not all(len(df) > 0 and len(df) == len(aligned_dfs[0]) for df in aligned_dfs): continue
            
            # --- 数据合并 ---
            # 按列（axis=1）将三个对齐后的数据表拼接成一个宽表
            combined_df = pd.concat(aligned_dfs, axis=1) 
            
            # 再次检查并确保列名正确
            if len(combined_df.columns) == len(ALL_FEATURE_COLUMNS):
                 combined_df.columns = ALL_FEATURE_COLUMNS
            else:
                 continue # 如果列数不匹配则跳过 

            # 如果合并后的数据长度不足一个序列窗口（4秒），则跳过
            if len(combined_df) < WINDOW_SIZE: continue 
            
            # --- 数据和标签存储 (*** 这是关键改动 ***) ---
            
            # 从 trial_key 中获取 subject_id
            subject_id = trial_key[0]
            
            # 将处理好的数据（转换为Numpy数组）存入列表
            trial_data = combined_df.values 
            # 根据元数据判断该试验是"FALLS"还是"ADL"，并存入标签（1代表跌倒，0代表非跌倒）
            trial_label = 1 if trial_metadata_map[trial_key]["category"] == "FALLS" else 0 
            
            # 将 (数据, 标签) 元组存入对应受试者的列表中
            subject_data_map[subject_id].append((trial_data, trial_label))
            
        except Exception:
            # 捕获任何在对齐和合并过程中可能出现的意外错误，并跳过该试验
            continue

    print(f"Successfully processed and grouped data for {len(subject_data_map)} subjects.")
    # 返回包含所有处理好的试验数据和标签的字典
    return subject_data_map

def create_sequences(data_list, label_list, seq_length, step):
    """使用滑动窗口从试验数据创建序列。"""
    # 初始化用于存放最终序列和对应标签的列表
    X, y = [], []
    # 遍历每一次活动试验的数据
    for i, trial_data in enumerate(data_list):
        trial_label = label_list[i]
        # 在单次试验数据上，按指定的步长（step）移动窗口
        for j in range(0, len(trial_data) - seq_length + 1, step):
            # 截取一个固定长度（seq_length）的片段作为序列
            X.append(trial_data[j:(j + seq_length)])
            # 为这个序列分配对应的标签
            y.append(trial_label)
            
    if not X: return np.array([]), np.array([])
    # 将列表转换为Numpy数组后返回
    return np.array(X), np.array(y)

# --- [添加这部分] ---

# 定义一个目录来存放按受试者保存的文件
OUTPUT_DIR = "Processed_Per_Subject"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载并按受试者分组数据
#    subject_data_map 的结构: {sub_id: [(trial_data_1, label_1), (trial_data_2, label_2), ...]}
subject_data_map = load_data_from_structured_folders(DATASET_PATH)

if not subject_data_map:
    print("No data was processed. Exiting.")
else:
    print(f"\nCreating sequences for {len(subject_data_map)} subjects...")
    
    total_sequences = 0
    
    # 2. 遍历每个受试者的数据
    for subject_id, data_label_list in subject_data_map.items():
        
        # 检查该受试者是否有有效试验
        if not data_label_list:
            print(f"No valid trials for subject {subject_id}. Skipping.")
            continue
            
        # 将 (data, label) 元组列表解压成两个单独的列表
        trial_arrays_for_subject = [item[0] for item in data_label_list]
        trial_labels_for_subject = [item[1] for item in data_label_list]

        # 3. 为该受试者创建序列
        X_subject, y_subject = create_sequences(
            trial_arrays_for_subject, 
            trial_labels_for_subject, 
            WINDOW_SIZE, 
            STEP
        )
        # 检查是否成功生成了序列
        if X_subject.size == 0:
            print(f"No sequences generated for subject {subject_id} (not enough data?). Skipping.")
            continue
            
        print(f"Subject {subject_id}: X_shape={X_subject.shape}, y_shape={y_subject.shape}")
        total_sequences += X_subject.shape[0]

        # 4. 定义特定于受试者的文件路径
        data_path = os.path.join(OUTPUT_DIR, f'SensorDataSequences_sub{subject_id}.npy')
        label_path = os.path.join(OUTPUT_DIR, f'SensorLabelSequences_sub{subject_id}.npy')
        
        # 5. 保存 .npy 文件
        np.save(data_path, X_subject)
        np.save(label_path, y_subject)
        
    print(f"\nAll per-subject data saved to '{OUTPUT_DIR}' directory.")
    print(f"Total sequences generated across all subjects: {total_sequences}")
