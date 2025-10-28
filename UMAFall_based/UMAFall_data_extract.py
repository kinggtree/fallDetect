import os
import numpy as np
import pandas as pd

# --- Configuration ---
# 包含所有 UMAFall CSV 文件的目录
DATASET_PATH = 'UMAFall_Dataset' 
TARGET_SAMPLING_RATE_HZ = 20.0  # 目标采样率 (Hz)
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms" # 采样周期 "50ms"
WINDOW_SECONDS = 4  # 窗口大小 (秒)
# 窗口大小 (采样点数) = 20Hz * 4s = 80
WINDOW_SIZE = int(TARGET_SAMPLING_RATE_HZ * WINDOW_SECONDS) 
STEP_SECONDS = 1 # 步长 (秒)
# 步长 (采样点数) = 20Hz * 1s = 20
STEP = int(TARGET_SAMPLING_RATE_HZ * STEP_SECONDS)          

# 传感器代码
SENSOR_CODES = ["acc", "gyro", "mag"]

# =============================================================================
#  配置: 根据你的信息
#  1. 我们只使用 "sensor id" 为 3 的数据
#  2. 使用 "sensor type" 列来区分传感器
# =============================================================================
TARGET_SENSOR_ID = 3 
SENSOR_TYPE_MAP = {
    0: 'acc',  # Accelerometer
    1: 'gyro', # Gyroscope
    2: 'mag'   # Magnetometer
}
# =============================================================================

# =============================================================================
#  CSV 列名顺序
#  TimeStamp(0); Sample No(1); X(2); Y(3); Z(4); Sensor Type(5); Sensor ID(6);
# =============================================================================
CSV_COLUMN_NAMES = ['timestamp', 'sample_no', 'x', 'y', 'z', 'sensor_type', 'sensor_id']
# =============================================================================

# 最终特征列
ALL_FEATURE_COLUMNS = [
    "acc_x", "acc_y", "acc_z", "acc_smv",
    "gyro_x", "gyro_y", "gyro_z", "gyro_smv",
    "mag_x", "mag_y", "mag_z" 
]

def load_all_trials(dataset_path):
    """
    加载 UMAFall 数据集中的所有试验。
    遍历 `dataset_path` 中的所有 CSV 文件,
    过滤 `sensor_id == 3` (第6列) 的数据,
    并根据 `sensor_type` (第5列) 
    (0=acc, 1=gyro, 2=mag) 提取数据。
    """
    trial_data_list = []
    trial_label_list = []
    
    print(f"开始扫描目录: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return [], []

    # =============================================================================
    #  !!! 变更 (Change) !!!
    #  对文件名进行排序，以确保按受试者顺序 (Subject_01, Subject_02, ...) 处理
    # =============================================================================
    try:
        all_filenames = os.listdir(dataset_path)
    except FileNotFoundError:
        print(f"错误: 找不到目录 {dataset_path}")
        return [], []
        
    sorted_filenames = sorted(all_filenames)
    print(f"找到了 {len(sorted_filenames)} 个文件。开始按字母顺序处理...")
    # =============================================================================

    for filename in sorted_filenames: # <-- 遍历排好序的列表
        if not filename.endswith('.csv'):
            continue
        
        file_path = os.path.join(dataset_path, filename)
        
        # 1. 从文件名确定标签
        if '_Fall_' in filename:
            label_id = 1 # 1 代表 "Fall"
            label_name = 'Fall'
        elif '_ADL_' in filename:
            label_id = 0 # 0 代表 "ADL" (非 Fall)
            label_name = 'ADL'
        else:
            continue # 不是可识别的试验文件

        # 2. 加载原始数据文件
        try:
            # 动态查找数据起始行（跳过所有 '%' 注释, 包括第41行的表头）
            data_start_line = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    stripped_line = line.strip()
                    if stripped_line.startswith('%') or not stripped_line:
                        data_start_line = i + 1
                    else:
                        break # 找到了第一行非空、非注释的数据
            
            raw_df = pd.read_csv(
                file_path,
                delimiter=';',
                skiprows=data_start_line,
                header=None,
                # 使用修正后的列名
                names=CSV_COLUMN_NAMES, 
                on_bad_lines='skip'
            )
        except Exception as e:
            print(f"警告: 无法读取文件 {filename}。错误: {e}")
            continue
        
        # 3. 过滤 "sensor id == 3" 的数据
        tag_df = raw_df[raw_df['sensor_id'] == TARGET_SENSOR_ID].copy()
        
        if tag_df.empty:
            # print(f"跳过文件 (没有 sensor_id == {TARGET_SENSOR_ID} 的数据): {filename}")
            continue
        
        # 4. 映射 "sensor_type" 到 "sensor_name"
        tag_df['sensor_name'] = tag_df['sensor_type'].map(SENSOR_TYPE_MAP)
        
        # 丢弃任何未在 SENSOR_TYPE_MAP 中的数据
        tag_df = tag_df.dropna(subset=['sensor_name'])
        
        if tag_df.empty:
            # print(f"警告: 跳过 {filename}。找到了 ID 3 但 'sensor_type' 不在 (0,1,2) 中。")
            continue
            
        all_sensors_df = []
        
        # 5. 为每个传感器生成时间戳、重采样并重命名列
        for sensor_name, group in tag_df.groupby('sensor_name'):
            group = group.copy()
            
            # 生成一个基于采样率的时间戳索引 (20Hz)
            time_delta_ms = 1000 / TARGET_SAMPLING_RATE_HZ
            group['timestamp_idx'] = pd.to_datetime(np.arange(len(group)) * time_delta_ms, unit='ms')
            group = group.set_index('timestamp_idx')
            
            # 选择并重命名 XYZ 列
            if sensor_name == 'acc':
                sensor_df = group[['x', 'y', 'z']].rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'})
            elif sensor_name == 'gyro':
                sensor_df = group[['x', 'y', 'z']].rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'})
            elif sensor_name == 'mag':
                sensor_df = group[['x', 'y', 'z']].rename(columns={'x': 'mag_x', 'y': 'mag_y', 'z': 'mag_z'})
            else:
                continue
                        
            # 重采样到统一的采样率 (TARGET_SAMPLING_PERIOD, e.g., "50ms")
            sensor_df = sensor_df.resample(TARGET_SAMPLING_PERIOD).mean()
            all_sensors_df.append(sensor_df)
        
        if not all_sensors_df:
            # print(f"警告: 跳过 {filename} (重采样后没有数据)。")
            continue
            
        # 6. 合并所有传感器数据
        try:
            combined_df = pd.concat(all_sensors_df, axis=1)
        except Exception as e:
            print(f"错误: 合并传感器失败 {filename}: {e}")
            continue

        # 7. 插值（填充重采样可能导致的缺失值）
        combined_df = combined_df.interpolate(method='linear', limit_direction='both', axis=0)
        
        # 8. 计算 SMV (Signal Magnitude Vector)
        try:
            if 'acc_x' in combined_df.columns:
                combined_df['acc_smv'] = np.sqrt(
                    combined_df['acc_x']**2 + 
                    combined_df['acc_y']**2 + 
                    combined_df['acc_z']**2
                )
            if 'gyro_x' in combined_df.columns:
                combined_df['gyro_smv'] = np.sqrt(
                    combined_df['gyro_x']**2 + 
                    combined_df['gyro_y']**2 + 
                    combined_df['gyro_z']**2
                )
        except Exception as e:
            print(f"警告: 计算 SMV 失败 {filename}: {e}")
            combined_df = combined_df.fillna(0) # 确保后续步骤没有NaN
            
        # 填充 SMV 计算中可能产生的 NaNs (如果 x,y,z 都是 NaN)
        combined_df = combined_df.fillna(0)

        # 9. 确保所有需要的列都存在
        try:
            # 按照 ALL_FEATURE_COLUMNS 定义的顺序选择和排序
            final_trial_df = combined_df[ALL_FEATURE_COLUMNS]
        except KeyError as e:
            print(f"警告: 跳过 {filename}。缺少必要的特征列: {e}。")
            print(f"    可用列: {list(combined_df.columns)}")
            print(f"    这通常意味着此文件缺少 sensor_type 0, 1, 或 2 (对于 sensor_id 3)。")
            continue
            
        # 10. 添加到列表
        trial_data_list.append(final_trial_df.to_numpy())
        trial_label_list.append(label_id)
        # 打印成功处理的文件名，让你能看到顺序
        print(f"成功处理: {filename} (标签: {label_name}, 形状: {final_trial_df.shape})")

    if not trial_data_list:
         print("错误: 没有加载任何数据。")
         print(f"请检查 DATASET_PATH ('{dataset_path}') 是否正确。")
         print("并确保所有文件中都包含 'sensor_id == 3' 的数据。")
         return [], []
         
    return trial_data_list, trial_label_list


def generate_sequences(data_list, label_list, seq_length, step):
    """
    (此函数来自你的原始脚本，无需修改)
    从加载的试验数据中生成固定长度的序列。
    """
    X, y = [], []
    # 遍历每一次活动试验的数据
    for i, trial_data in enumerate(data_list):
        trial_label = label_list[i]
        
        if len(trial_data) < seq_length:
            print(f"警告: 第 {i} 个试验数据长度 ({len(trial_data)}) 小于窗口大小 ({seq_length})，已跳过。")
            continue
            
        # 在单次试验数据上，按指定的步长（step）移动窗口
        for j in range(0, len(trial_data) - seq_length + 1, step):
            # 截取一个固定长度（seq_length）的片段作为序列
            X.append(trial_data[j:(j + seq_length)])
            # 为这个序列分配对应的标签
            y.append(trial_label)
            
    if not X: 
        print("错误: 未能生成任何序列。")
        return np.array([]), np.array([])
        
    # 将列表转换为Numpy数组后返回
    return np.array(X), np.array(y)

# --- Main execution ---

SensorDataSequences, SensorLabelSequences = np.array([]), np.array([])

# 定义输出文件名 (更新以反映新逻辑)
DATASEQ_PATH = f'SensorDataSequences_UMAFall_ID3_20Hz_{WINDOW_SECONDS}s.npy'
LABELSEQ_PATH = f'SensorLabelSequences_UMAFall_ID3_20Hz_{WINDOW_SECONDS}s.npy'

if os.path.exists(DATASEQ_PATH) and os.path.exists(LABELSEQ_PATH):
    print("找到了已存在的 npy 文件。正在加载...")
    SensorDataSequences = np.load(DATASEQ_PATH)
    print(f"加载数据集形状: X={SensorDataSequences.shape}")
    SensorLabelSequences = np.load(LABELSEQ_PATH)
    print(f"加载数据集形状: y={SensorLabelSequences.shape}")
else:
    print("未找到 npy 文件。开始处理原始 CSV 数据...")
    
    # 1. 加载所有试验数据
    trial_data_list, trial_label_list = load_all_trials(DATASET_PATH)
    
    if trial_data_list:
        print(f"\n成功加载 {len(trial_data_list)} 个试验。")
        
        # 2. 生成序列
        print(f"正在生成序列... 窗口大小={WINDOW_SIZE}, 步长={STEP}")
        SensorDataSequences, SensorLabelSequences = generate_sequences(
            trial_data_list, 
            trial_label_list, 
            seq_length=WINDOW_SIZE, 
            step=STEP
        )
        
        if SensorDataSequences.size > 0:
            print(f"\n序列生成完毕。")
            print(f"最终数据集形状: X={SensorDataSequences.shape}") # 应该为 (xxxx, 120, 11)
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