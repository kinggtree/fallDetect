import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import io

# --- Configuration ---
DATASET_PATH = 'UMAFall_Dataset' 
TARGET_SAMPLING_RATE_HZ = 20.0  # 目标采样率 (Hz)
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms" # "50ms"
WINDOW_SECONDS = 6
WINDOW_SIZE = int(TARGET_SAMPLING_RATE_HZ * WINDOW_SECONDS) # 120
STEP_SECONDS = 2
STEP = int(TARGET_SAMPLING_RATE_HZ * STEP_SECONDS) # 40

# =============================================================================
#  !!! 变更 (Change) !!!
#  1. 目标 Sensor ID 列表
#  2. 新增 Sensor ID 0 (手机) 的配置
# =============================================================================
TARGET_SENSOR_IDS = [1, 2, 3, 4] # (Tag) 4 个 11 特征的传感器
SPECIAL_SENSOR_ID = 0           # (Phone) 1 个 4 特征的传感器
SPECIAL_SENSOR_HZ = 200.0       # (Phone) 采样率 200Hz

SENSOR_TYPE_MAP = {
    0: 'acc',  # Accelerometer
    1: 'gyro', # Gyroscope
    2: 'mag'   # Magnetometer
}
CSV_COLUMN_NAMES = ['timestamp', 'sample_no', 'x', 'y', 'z', 'sensor_type', 'sensor_id']

# (11) 个主要特征 (用于 ID 1, 2, 3, 4)
MAIN_FEATURE_COLUMNS = [
    "acc_x", "acc_y", "acc_z", "acc_smv",
    "gyro_x", "gyro_y", "gyro_z", "gyro_smv",
    "mag_x", "mag_y", "mag_z" 
]
# (4) 个特殊特征 (用于 ID 0)
SPECIAL_SENSOR_FEATURES = [
    "acc_x", "acc_y", "acc_z", "acc_smv"
]

# (48) 个最终特征 (4 * 11 + 4 = 48)
# e.g., ['acc_x_s1', ..., 'mag_z_s4', 'acc_x_s0', ..., 'acc_smv_s0']
COLS_MAIN = [
    f"{col}_s{sid}" 
    for sid in TARGET_SENSOR_IDS 
    for col in MAIN_FEATURE_COLUMNS
]
COLS_SPECIAL = [
    f"{col}_s{SPECIAL_SENSOR_ID}" 
    for col in SPECIAL_SENSOR_FEATURES
]
FINAL_COLUMN_ORDER = COLS_MAIN + COLS_SPECIAL
# =============================================================================


def process_sensor_group(tag_df, feature_columns, sensor_id_debug):
    """
    (函数与上一版相同)
    辅助函数：处理单个 sensor_id (1, 2, 3, 4) 的数据。
    输入: 原始 DataFrame (已按 sensor_id 过滤), 特征列表 (11)
    输出: 处理后的 DataFrame (N, 11) 或 None
    """
    tag_df = tag_df.copy()
    
    # 1. 映射 "sensor_type" (0,1,2) 到 "sensor_name" (acc,gyro,mag)
    tag_df['sensor_name'] = tag_df['sensor_type'].map(SENSOR_TYPE_MAP)
    tag_df = tag_df.dropna(subset=['sensor_name'])
    
    if tag_df.empty:
        # print(f"    警告: (ID {sensor_id_debug}) 'sensor_type' 不在 (0,1,2) 中。")
        return None
        
    all_sensors_df = []
    
    # 2. 为每个传感器(acc, gyro, mag)分别处理
    sensor_names_found = set()
    for sensor_name, group in tag_df.groupby('sensor_name'):
        sensor_names_found.add(sensor_name)
        group = group.copy()
        
        # 2a. 生成时间戳索引 (20Hz)
        time_delta_ms = 1000 / TARGET_SAMPLING_RATE_HZ
        group['timestamp_idx'] = pd.to_datetime(np.arange(len(group)) * time_delta_ms, unit='ms')
        group = group.set_index('timestamp_idx')
        
        # 2b. 重命名 (x,y,z) -> (acc_x, acc_y, acc_z) 等
        sensor_df = group[['x', 'y', 'z']].rename(columns={
            'x': f'{sensor_name}_x', 
            'y': f'{sensor_name}_y', 
            'z': f'{sensor_name}_z'
        })
                    
        # 2c. 重采样以对齐 (20Hz -> 20Hz, 确保时间点一致)
        sensor_df = sensor_df.resample(TARGET_SAMPLING_PERIOD).mean()
        all_sensors_df.append(sensor_df)
    
    # 3. 检查是否所有 3 个传感器都存在
    if 'acc' not in sensor_names_found or 'gyro' not in sensor_names_found or 'mag' not in sensor_names_found:
        # print(f"    警告: (ID {sensor_id_debug}) 缺少 acc, gyro, mag 之一。跳过。")
        return None
        
    # 4. 合并 (acc, gyro, mag) -> (N, 9)
    try:
        combined_df = pd.concat(all_sensors_df, axis=1)
    except Exception as e:
        # print(f"    错误: (ID {sensor_id_debug}) 合并传感器失败: {e}")
        return None

    # 5. 插值（填充缺失值）
    combined_df = combined_df.interpolate(method='linear', limit_direction='both', axis=0)
    
    # 6. 计算 SMV -> (N, 11)
    try:
        combined_df['acc_smv'] = np.sqrt(
            combined_df['acc_x']**2 + combined_df['acc_y']**2 + combined_df['acc_z']**2
        )
        combined_df['gyro_smv'] = np.sqrt(
            combined_df['gyro_x']**2 + combined_df['gyro_y']**2 + combined_df['gyro_z']**2
        )
    except Exception:
        combined_df = combined_df.fillna(0)
    
    combined_df = combined_df.fillna(0)

    # 7. 确保所有 (11) 列都存在
    try:
        final_df = combined_df[feature_columns]
    except KeyError:
        # print(f"    警告: (ID {sensor_id_debug}) 缺少特征列。")
        return None
        
    return final_df


def process_special_sensor(phone_df, feature_columns, sensor_id_debug):
    """
    !!! 新函数 !!!
    辅助函数：处理 Sensor ID 0 (200Hz 手机) 的数据。
    输入: 原始 DataFrame (已按 sensor_id 过滤), 特征列表 (4)
    输出: 处理后的 DataFrame (N, 4) 或 None
    """
    phone_df = phone_df.copy()

    # 1. 只获取 Accelerometer (sensor_type == 0)
    acc_df = phone_df[phone_df['sensor_type'] == 0].copy()
    
    if acc_df.empty:
        # print(f"    警告: (ID {sensor_id_debug}) 缺少 sensor_type 0 (Accelerometer) 数据。")
        return None
        
    # 2. 生成时间戳索引 (使用 200Hz 采样率)
    time_delta_ms = 1000 / SPECIAL_SENSOR_HZ # (5ms)
    acc_df['timestamp_idx'] = pd.to_datetime(np.arange(len(acc_df)) * time_delta_ms, unit='ms')
    acc_df = acc_df.set_index('timestamp_idx')

    # 3. 重命名
    acc_df = acc_df[['x', 'y', 'z']].rename(columns={
        'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'
    })
    
    # 4. !!! 关键: 降采样 (Downsample) 200Hz -> 20Hz !!!
    #    (e.g., "5ms" -> "50ms")
    acc_df_resampled = acc_df.resample(TARGET_SAMPLING_PERIOD).mean()
    
    # 5. 插值
    acc_df_resampled = acc_df_resampled.interpolate(method='linear', limit_direction='both', axis=0)

    # 6. 计算 SMV (在 20Hz 的数据上)
    try:
        acc_df_resampled['acc_smv'] = np.sqrt(
            acc_df_resampled['acc_x']**2 + 
            acc_df_resampled['acc_y']**2 + 
            acc_df_resampled['acc_z']**2
        )
    except Exception:
        acc_df_resampled = acc_df_resampled.fillna(0)
        
    acc_df_resampled = acc_df_resampled.fillna(0)

    # 7. 确保所有 (4) 列都存在
    try:
        final_df = acc_df_resampled[feature_columns]
    except KeyError:
        # print(f"    警告: (ID {sensor_id_debug}) 缺少特征列。")
        return None

    return final_df
    

def load_all_trials(dataset_path):
    """
    (重构)
    加载 UMAFall 数据集中的所有试验。
    - 遍历 CSV 文件
    - 对每个文件，提取 ID 1, 2, 3, 4 (Tag) 和 0 (Phone)
    - 单独处理 (1,2,3,4) -> (N, 11)
    - 单独处理 (0) 并降采样 -> (N, 4)
    - 截断到 5 个流中的最短长度
    - 合并为 (min_len, 48) 的试验
    """
    trial_data_list = []
    trial_label_list = []
    
    print(f"开始扫描目录: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return [], []

    sorted_filenames = sorted(os.listdir(dataset_path))
    print(f"找到了 {len(sorted_filenames)} 个文件。开始按字母顺序处理...")

    for filename in sorted_filenames: 
        if not filename.endswith('.csv'):
            continue
        
        file_path = os.path.join(dataset_path, filename)
        
        # 1. 从文件名确定标签
        if '_Fall_' in filename:
            label_id = 1
            label_name = 'Fall'
        elif '_ADL_' in filename:
            label_id = 0
            label_name = 'ADL'
        else:
            continue

        # 2. 加载原始数据文件
        try:
            data_start_line = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    stripped_line = line.strip()
                    if stripped_line.startswith('%') or not stripped_line:
                        data_start_line = i + 1
                    else:
                        break
            
            raw_df = pd.read_csv(
                file_path, delimiter=';', skiprows=data_start_line,
                header=None, names=CSV_COLUMN_NAMES, on_bad_lines='skip'
            )
        except Exception as e:
            print(f"警告: 无法读取文件 {filename}。错误: {e}")
            continue
        
        # 3. 按 'sensor_id' 分组
        groups_by_id = raw_df.groupby('sensor_id')
        processed_main_dfs = []   # (List of 4 DataFrames)
        processed_special_df = None # (Single DataFrame)
        
        # 4. 遍历我们需要的 4 个主要 ID (1, 2, 3, 4)
        valid_file = True
        for sensor_id in TARGET_SENSOR_IDS:
            if sensor_id not in groups_by_id.groups:
                # print(f"警告: {filename} 缺少 sensor_id {sensor_id}。跳过此文件。")
                valid_file = False
                break
            
            df_group = groups_by_id.get_group(sensor_id)
            
            # 5. 调用 'process_sensor_group' (返回 11 特征)
            processed_df = process_sensor_group(
                df_group, 
                MAIN_FEATURE_COLUMNS, 
                sensor_id_debug=sensor_id
            )
            
            if processed_df is None:
                print(f"警告: {filename} 中 sensor_id {sensor_id} 的数据处理失败。跳过此文件。")
                valid_file = False
                break
            
            processed_main_dfs.append(processed_df)

        if not valid_file:
            continue
            
        # 6. 处理特殊 ID (0)
        if SPECIAL_SENSOR_ID not in groups_by_id.groups:
            # print(f"警告: {filename} 缺少 sensor_id {SPECIAL_SENSOR_ID} (Phone)。跳过此文件。")
            continue
            
        df_group = groups_by_id.get_group(SPECIAL_SENSOR_ID)
        
        # 7. 调用 'process_special_sensor' (返回 4 特征)
        processed_special_df = process_special_sensor(
            df_group,
            SPECIAL_SENSOR_FEATURES,
            sensor_id_debug=SPECIAL_SENSOR_ID
        )

        if processed_special_df is None:
            print(f"警告: {filename} 中 sensor_id {SPECIAL_SENSOR_ID} 的数据处理失败。跳过此文件。")
            continue

        # 8. 截断: 找到 5 个 (4+1) DataFrame 中最短的长度
        all_dfs = processed_main_dfs + [processed_special_df]
        try:
            min_len = min(len(df) for df in all_dfs)
            if min_len < WINDOW_SIZE:
                print(f"警告: {filename} 的最短数据长度 ({min_len}) 小于窗口大小 ({WINDOW_SIZE})。跳过。")
                continue
        except ValueError:
            continue # 列表为空
            
        # 9. 合并 (Concat)
        final_dfs_to_concat = []
        
        # 9a. 添加 4 个主要 sensor (重命名: _s1, _s2, ...)
        for i, df in enumerate(processed_main_dfs):
            sensor_id = TARGET_SENSOR_IDS[i]
            df_truncated = df.iloc[:min_len]
            df_renamed = df_truncated.rename(columns=lambda col: f"{col}_s{sensor_id}")
            final_dfs_to_concat.append(df_renamed)
            
        # 9b. 添加 1 个特殊 sensor (重命名: _s0)
        df_special_truncated = processed_special_df.iloc[:min_len]
        df_special_renamed = df_special_truncated.rename(columns=lambda col: f"{col}_s{SPECIAL_SENSOR_ID}")
        final_dfs_to_concat.append(df_special_renamed)
            
        # 9c. 沿列 (axis=1) 合并
        final_trial_df = pd.concat(final_dfs_to_concat, axis=1)
        
        # 10. 确保列顺序是我们期望的 (48 列)
        final_trial_df = final_trial_df[FINAL_COLUMN_ORDER]
            
        # 11. 添加到列表
        trial_data_list.append(final_trial_df.to_numpy())
        trial_label_list.append(label_id)
        print(f"成功处理: {filename} (标签: {label_name}, 形状: {final_trial_df.shape})") # (min_len, 48)

    if not trial_data_list:
         print("错误: 没有加载任何数据。")
         print(f"请检查 DATASET_PATH ('{dataset_path}') 是否正确。")
         print(f"并确保文件 *同时* 包含 sensor_id {TARGET_SENSOR_IDS} + [{SPECIAL_SENSOR_ID}] 的数据。")
         return [], []
         
    return trial_data_list, trial_label_list


def generate_sequences(data_list, label_list, seq_length, step):
    """
    (此函数无需修改)
    从加载的试验数据中生成固定长度的序列。
    输入: data_list (list of (N, 48) arrays)
    输出: (xxxx, 120, 48)
    """
    X, y = [], []
    for i, trial_data in enumerate(data_list): # trial_data 是 (N, 48)
        trial_label = label_list[i]
        
        if len(trial_data) < seq_length:
            print(f"警告: 第 {i} 个试验数据长度 ({len(trial_data)}) 小于窗口大小 ({seq_length})，已跳过。")
            continue
            
        for j in range(0, len(trial_data) - seq_length + 1, step):
            # X.append 截取的是 (seq_length, 48)
            X.append(trial_data[j:(j + seq_length)]) 
            y.append(trial_label)
            
    if not X: 
        print("错误: 未能生成任何序列。")
        return np.array([]), np.array([])
        
    return np.array(X), np.array(y)

# --- Main execution ---

SensorDataSequences, SensorLabelSequences = np.array([]), np.array([])

# 变更: 更新输出文件名
DATASEQ_PATH = f'SensorDataSequences_UMAFall_IDs1234and0_20Hz_{WINDOW_SECONDS}s.npy'
LABELSEQ_PATH = f'SensorLabelSequences_UMAFall_IDs1234and0_20Hz_{WINDOW_SECONDS}s.npy'

if os.path.exists(DATASEQ_PATH) and os.path.exists(LABELSEQ_PATH):
    print("找到了已存在的 npy 文件。正在加载...")
    SensorDataSequences = np.load(DATASEQ_PATH)
    print(f"加载数据集形状: X={SensorDataSequences.shape}")
    SensorLabelSequences = np.load(LABELSEQ_PATH)
    print(f"加载数据集形状: y={SensorLabelSequences.shape}")
else:
    print("未找到 npy 文件。开始处理原始 CSV 数据...")
    
    # 1. 加载所有试验数据 (现在返回 (N, 48) 形状的试验)
    trial_data_list, trial_label_list = load_all_trials(DATASET_PATH)
    
    if trial_data_list:
        print(f"\n成功加载 {len(trial_data_list)} 个试验。")
        
        # 2. 生成序列 (seq_length=120, step=40)
        print(f"正在生成序列... 窗口大小={WINDOW_SIZE}, 步长={STEP}")
        SensorDataSequences, SensorLabelSequences = generate_sequences(
            trial_data_list, 
            trial_label_list, 
            seq_length=WINDOW_SIZE, 
            step=STEP
        )
        
        if SensorDataSequences.size > 0:
            print(f"\n序列生成完毕。")
            # 最终形状应为 (xxxx, 120, 48)
            print(f"最终数据集形状: X={SensorDataSequences.shape}") 
            print(f"最终数据集形状: y={SensorLabelSequences.shape}")

            # 3. 保存为 .npy 文件
            print(f"正在保存到 {DATASEQ_PATH} 和 {LABELSEQ_PATH}...")
            np.save(DATASEQ_PATH, SensorDataSequences)
            np.save(LABELSEQ_PATH, SensorLabelSequences)
            print("保存完毕。")
        else:
            print("未能从加载的数据中生成任何序列。")
    else:
        print("未能加载任何试验数据，程序终止。")