import socket
import time
import numpy as np
import pickle
import struct
import pandas as pd
import os
import io
import re
from collections import defaultdict


from shared_config import HOST, PORT, SENSOR_CODES, TARGET_SAMPLING_RATE_HZ, ALL_FEATURE_COLUMNS, TARGET_SAMPLING_PERIOD, EXPECTED_COLUMNS, SEQUENCE_LENGTH, STEP, DATASET_PATH


class DataCleaner:
    """
    一个用于加载、清洗、对齐和序列化传感器数据的类。
    """
    def __init__(self, sequence_length, step):
        """
        初始化DataCleaner实例。
        
        Args:
            sequence_length (int): 每个数据序列的长度 (窗口大小)。
            step (int): 创建序列时滑动窗口的步长。
        """
        self.sequence_length = sequence_length
        self.step = step
        print(f"Data Cleaner initialized with sequence_length={self.sequence_length}, step={self.step}.")

    # --- 2. 公共调用接口 ---
    def process_directory(self, dataset_root_path):
        """
        处理整个数据集目录的入口方法。
        它按顺序执行加载和序列化两个步骤。
        
        Args:
            dataset_root_path (str): 数据集根目录的路径。
            
        Returns:
            tuple: 包含两个numpy数组 (X_sequences, y_sequences)。
        """
        print("Starting data processing pipeline...")
        # 调用内部方法加载和合并所有试验数据
        trial_arrays, trial_labels = self._load_data_from_structured_folders(dataset_root_path)
        
        if not trial_arrays:
            print("No trials were processed. Returning empty arrays.")
            return np.array([]), np.array([])
            
        # 基于加载的数据创建滑动窗口序列
        X_sequences, y_sequences = self._create_sequences(trial_arrays, trial_labels)
        print("Data processing pipeline finished.")
        return X_sequences, y_sequences

    # --- 3. 内部处理方法 (使用下划线_开头，表示建议内部使用) ---
    
    # 标记为静态方法，因为它不依赖于任何实例状态 (self)
    @staticmethod
    def _load_and_resample_sensor_file(filepath, sensor_code):
        """加载单个传感器文件，转换时间戳并进行重采样。"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            data_start_line_index = -1
            for i, line in enumerate(lines):
                if line.strip().upper() == "@DATA":
                    data_start_line_index = i + 1
                    break

            if data_start_line_index == -1 or data_start_line_index >= len(lines): return None

            data_string = "".join(lines[data_start_line_index:])
            if not data_string.strip(): return None

            df = pd.read_csv(io.StringIO(data_string), header=None, usecols=[0, 1, 2, 3])
            if df.empty: return None

            df.columns = ['timestamp_ns'] + EXPECTED_COLUMNS[sensor_code]
            df['timestamp'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
            df = df.set_index('timestamp').drop(columns=['timestamp_ns']).sort_index()
            
            # 使用全局定义的常量 TARGET_SAMPLING_PERIOD
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
            print(f"Error processing file {filepath}: {e}. Skipping.")
            return None

    # 这是实例方法，因为它需要访问 self.sequence_length
    def _load_data_from_structured_folders(self, dataset_root_path):
        """遍历数据集文件夹，处理、对齐并组合每个试验的传感器数据。"""
        print(f"Scanning for data in: {dataset_root_path}")
        if not os.path.isdir(dataset_root_path):
            print(f"ERROR: Dataset root path '{dataset_root_path}' not found.")
            return [], []

        trial_sensor_files_map = defaultdict(lambda: defaultdict(str))
        trial_metadata_map = {}
        
        for dirpath, _, filenames in os.walk(dataset_root_path):
            relative_path = os.path.relpath(dirpath, dataset_root_path)
            path_parts = relative_path.split(os.sep)
            if len(path_parts) != 3: continue

            for filename in filenames:
                if not filename.endswith(".txt"): continue
                fname_parts = filename.replace('.txt', '').split('_')
                if len(fname_parts) != 4: continue
                
                _, sensor_code, _, trial_no_str = fname_parts
                sensor_code = sensor_code.lower()
                if sensor_code not in SENSOR_CODES: continue

                try:
                    subject_match = re.fullmatch(r'sub(\d+)', path_parts[0], re.IGNORECASE)
                    if not subject_match: continue
                    subject_id = int(subject_match.group(1))
                    
                    category = path_parts[1].upper()
                    activity_code = path_parts[2].upper()
                    trial_no = int(trial_no_str)
                    filepath = os.path.join(dirpath, filename)
                    
                    trial_key = (subject_id, activity_code, trial_no)
                    trial_sensor_files_map[trial_key][sensor_code] = filepath
                    if trial_key not in trial_metadata_map:
                        trial_metadata_map[trial_key] = {"category": category, "activity_code": activity_code}
                except (AttributeError, ValueError):
                    continue

        processed_trials_data, labels = [], []
        print(f"\nProcessing and combining {len(trial_sensor_files_map)} unique trials...")
        
        for trial_key, sensor_files in trial_sensor_files_map.items():
            if not all(s_code in sensor_files for s_code in SENSOR_CODES): continue

            # 调用静态方法，注意要用 ClassName._method() 或 self._method()
            resampled_dfs = {s_code: self._load_and_resample_sensor_file(sensor_files[s_code], s_code) for s_code in SENSOR_CODES}
            if any(df is None or df.empty for df in resampled_dfs.values()): continue

            try:
                common_start = max(df.index.min() for df in resampled_dfs.values())
                common_end = min(df.index.max() for df in resampled_dfs.values())
                if common_start >= common_end: continue

                aligned_dfs = [resampled_dfs[s_code][common_start:common_end].reset_index(drop=True) for s_code in SENSOR_CODES]
                if not all(len(df) > 0 and len(df) == len(aligned_dfs[0]) for df in aligned_dfs): continue
                
                combined_df = pd.concat(aligned_dfs, axis=1)
                
                # 这里必须使用全局常量 ALL_FEATURE_COLUMNS 来检查和重命名
                expected_cols_count = len(ALL_FEATURE_COLUMNS)
                actual_cols_count = len(combined_df.columns)
                
                # 有时smv可能计算失败，导致列数不匹配，这里做一个兼容
                if actual_cols_count == expected_cols_count:
                    combined_df.columns = ALL_FEATURE_COLUMNS
                # 如果缺少smv列，我们只保留共有的部分
                elif actual_cols_count == expected_cols_count - 2: 
                    print(f"Warning: Trial {trial_key} is missing SMV columns. Proceeding without them.")
                    combined_df.columns = [c for c in ALL_FEATURE_COLUMNS if 'smv' not in c]
                else:
                    print(f"Warning: Trial {trial_key} column count ({actual_cols_count}) mismatch. Expected around {expected_cols_count}. Skipping.")
                    continue

                # 使用实例属性 self.sequence_length
                if len(combined_df) < self.sequence_length: continue
                
                processed_trials_data.append(combined_df.to_numpy()) # 推荐使用 .to_numpy()
                labels.append(1 if trial_metadata_map[trial_key]["category"] == "FALLS" else 0)
            except Exception as e:
                print(f"An error occurred while aligning trial {trial_key}: {e}")
                continue

        print(f"Successfully processed and combined sensor data for {len(processed_trials_data)} trials.")
        return processed_trials_data, labels

    # 这是实例方法，因为它需要访问 self.sequence_length 和 self.step
    def _create_sequences(self, data_list, label_list):
        """使用滑动窗口从试验数据创建序列。"""
        X, y = [], []
        for i, trial_data in enumerate(data_list):
            trial_label = label_list[i]
            # 使用实例属性 self.sequence_length 和 self.step
            for j in range(0, len(trial_data) - self.sequence_length + 1, self.step):
                X.append(trial_data[j:(j + self.sequence_length)])
                y.append(trial_label)
                
        if not X: return np.array([]), np.array([])
        return np.array(X), np.array(y)




def get_data_stream(sequences_array):
    """从序列数据重建连续数据流的生成器。"""
    continuous_data = sequences_array[0]
    additional_points = sequences_array[1:, -1, :]
    continuous_data = np.vstack((continuous_data, additional_points))
    
    for data_point in continuous_data:
        yield data_point

def main():
    print("Starting Data Provider...")
    
    # --- 1. 加载数据并清洗 ---
    # (1) 用配置参数（窗口长度和步长）来实例化这个类
    cleaner = DataCleaner(sequence_length=SEQUENCE_LENGTH, step=STEP)

    # (2) 调用主方法来处理整个目录，它会返回最终的序列化数据
    X_sequences, _ = cleaner.process_directory(DATASET_PATH)

    data_stream = get_data_stream(X_sequences)
    
    # --- 3. 设置网络连接 (作为客户端) ---
    print(f"Attempting to connect to Edge Inference Node at {HOST}:{PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Connection established. Starting data transmission...")
        
        # 模拟50Hz的采样率
        sampling_period = 1.0 / TARGET_SAMPLING_RATE_HZ
        
        for raw_point in data_stream:
            start_time = time.time()
            
            # 清洗数据
            cleaned_point = raw_point
            
            # 序列化数据以便网络传输
            data_bytes = pickle.dumps(cleaned_point)
            
            # 创建消息头 (包含消息长度)，然后发送
            # 'I' 代表一个4字节的无符号整数
            header = struct.pack('!I', len(data_bytes))
            s.sendall(header + data_bytes)
            
            # 控制发送速率
            time_to_sleep = sampling_period - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
                
    print("Data transmission complete. Connection closed.")

if __name__ == '__main__':
    main()