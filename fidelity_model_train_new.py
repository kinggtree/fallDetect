import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import io
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset


# --- Configuration ---
DATASET_PATH = 'MobiFall_Dataset'
TARGET_SAMPLING_RATE_HZ = 50.0  # Target sampling rate in Hz
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms"
WINDOW_SECONDS = 4
WINDOW_SIZE = int(TARGET_SAMPLING_RATE_HZ * WINDOW_SECONDS) # 200 samples for 4 seconds at 50Hz

STEP_SECONDS = 1 # 1秒步长
STEP = int(TARGET_SAMPLING_RATE_HZ * STEP_SECONDS)          # 50 samples for 1 second step at 50Hz

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
    """遍历数据集文件夹，处理、对齐并组合每个试验的传感器数据。"""
    print(f"Scanning for data in: {dataset_root_path}")
    if not os.path.isdir(dataset_root_path):
        print(f"ERROR: Dataset root path '{dataset_root_path}' not found.")
        return [], []

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

    # 初始化两个列表，用于存放最终处理好的数据和对应的标签
    processed_trials_data, labels = [], []
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
            
            # --- 数据和标签存储 ---
            # 将处理好的数据（转换为Numpy数组）存入列表
            processed_trials_data.append(combined_df.values)
            # 根据元数据判断该试验是"FALLS"还是"ADL"，并存入标签（1代表跌倒，0代表非跌倒）
            labels.append(1 if trial_metadata_map[trial_key]["category"] == "FALLS" else 0)
            
        except Exception:
            # 捕获任何在对齐和合并过程中可能出现的意外错误，并跳过该试验
            continue

    print(f"Successfully processed and combined sensor data for {len(processed_trials_data)} trials.")
    # 返回包含所有处理好的试验数据和标签的列表
    return processed_trials_data, labels

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

class FeatureModel1DCNN(nn.Module):
    def __init__(self, input_channels=11, num_classes=1):
        super(FeatureModel1DCNN, self).__init__()
        
        # 特征提取器: 包含一系列的卷积和池化层
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2), # Length: 200 -> 100
            
            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2), # Length: 100 -> 50

            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Length: 50 -> 25
        )
        
        # 分类器: 将提取的特征映射到最终的输出
        # 输入维度需要计算: 256 (channels) * 25 (length)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 25, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        标准的前向传播，用于训练和评估
        x 的输入形状: (batch_size, sequence_length, num_features) -> (N, 200, 11)
        """
        # Conv1d 需要 (N, C, L) 格式, 所以我们需要重排维度
        x = x.permute(0, 2, 1) # -> (N, 11, 200)
        
        features = self.feature_extractor(x)
        output = self.classifier(features)
        
        # 因为使用 BCEWithLogitsLoss, 所以不需要在这里加 sigmoid
        return output

    def extract_features(self, x):
        """
        仅用于提取中间特征的函数
        x 的输入形状: (batch_size, sequence_length, num_features) -> (N, 200, 11)
        """
        # 同样需要重排维度
        x = x.permute(0, 2, 1) # -> (N, 11, 200)
        
        # 只通过特征提取器
        features = self.feature_extractor(x)
        
        # 输出形状将是 (N, 256, 25)
        return features

def create_sparse_data(data_array, sparsity_ratio):
    """
    Randomly sets a portion of samples in a data array to zero.

    Args:
        data_array (np.ndarray): The input data array, e.g., shape (9491, 200, 11).
        sparsity_ratio (float): The fraction of samples to set to zero (between 0.0 and 1.0).

    Returns:
        np.ndarray: A new data array with the specified portion of samples zeroed out.
    """
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError("Sparsity ratio must be between 0.0 and 1.0")

    # 创建一个副本以避免修改原始数组
    sparse_array = data_array.copy()
    
    # 获取样本总数
    num_samples = sparse_array.shape[0]
    
    # 计算需要置零的样本数量
    num_to_zero_out = int(num_samples * sparsity_ratio)
    
    if num_to_zero_out == 0:
        print("Sparsity ratio is too low, no samples will be zeroed out.")
        return sparse_array

    # 随机选择不重复的索引进行置零
    indices_to_zero = np.random.choice(
        np.arange(num_samples), 
        size=num_to_zero_out, 
        replace=False
    )
    
    # 将选定索引对应的整个 (200, 11) 向量置零
    sparse_array[indices_to_zero] = 0
    
    print(f"Sparsification complete:")
    print(f"  - Total samples: {num_samples}")
    print(f"  - Sparsity ratio: {sparsity_ratio:.2f}")
    print(f"  - Samples zeroed out: {len(indices_to_zero)}")
    
    return sparse_array


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x 的形状: (batch_size, time_steps, C, H, W) 或 (batch_size, time_steps, features...)
        # 我们这里是 (batch_size, 60, 200, 11)
        
        batch_size, time_steps = x.size(0), x.size(1)
        
        # 1. 合并 batch 和 time 维度
        # (B, T, C, F) -> (B * T, C, F)
        # 我们的输入是 (B, 60, 200, 11)，需要先 permute
        x = x.permute(0, 1, 3, 2) # -> (B, 60, 11, 200)
        x_reshape = x.contiguous().view(batch_size * time_steps, x.size(2), x.size(3))
        # -> (B * 60, 11, 200)

        # 2. 应用模块
        y = self.module(x_reshape)
        
        # y 的形状是 (B * 60, output_features)
        
        # 3. 恢复 batch 和 time 维度
        y = y.view(batch_size, time_steps, y.size(-1))
        # -> (B, 60, output_features)
        
        return y
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim)
        self.key_layer = nn.Linear(key_dim, hidden_dim)
        self.value_layer = nn.Linear(key_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, key, value):
        # query (来自LFS): (Batch, SeqLen, query_dim)
        # key/value (来自HFS): (Batch, SeqLen, key_dim)
        
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 应用权重
        context_vector = torch.matmul(attention_weights, V)
        return context_vector


def create_raw_data_cnn():
    """创建一个用于处理原始传感器数据的1D-CNN模块。"""
    raw_data_processor = nn.Sequential(
        nn.Conv1d(in_channels=11, out_channels=64, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm1d(64),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm1d(128),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'), nn.ReLU(), nn.BatchNorm1d(256),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten()
    )
    return raw_data_processor


class ContextualFidelityModel(nn.Module):
    def __init__(self, feature_dim, lstm_hidden_dim, raw_cnn_output_dim, num_classes=1):
        super(ContextualFidelityModel, self).__init__()

        # --- 分支一：高保真原始数据流处理器 ---
        raw_cnn = create_raw_data_cnn()
        self.hfs_processor = TimeDistributed(raw_cnn)

        # --- 分支二：低保真特征流处理器 ---
        self.lfs_processor = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        # --- 融合模块：交叉注意力 ---
        # query 来自 lfs_processor (lstm_hidden_dim)
        # key/value 来自 hfs_processor (raw_cnn_output_dim)
        self.cross_attention = CrossAttention(
            query_dim=lstm_hidden_dim,
            key_dim=raw_cnn_output_dim,
            hidden_dim=lstm_hidden_dim # 通常设置为与query_dim一致
        )
        
        # --- 后融合处理器与分类器 ---
        # 将 LSTM 的输出和注意力机制的输出结合起来
        self.post_fusion_processor = nn.LSTM(
            input_size=lstm_hidden_dim * 2, # Concatenated input
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, feature_sequence, imputed_raw_sequence):
        # feature_sequence: (B, 60, 6400)
        # imputed_raw_sequence: (B, 60, 200, 11)

        # 1. 并行处理两条流
        lfs_output, _ = self.lfs_processor(feature_sequence) # -> (B, 60, lstm_hidden_dim)
        hfs_output = self.hfs_processor(imputed_raw_sequence) # -> (B, 60, raw_cnn_output_dim)

        # 2. 交叉注意力融合
        # lfs_output 作为 Query，去查询 hfs_output
        attention_context = self.cross_attention(
            query=lfs_output, 
            key=hfs_output, 
            value=hfs_output
        ) # -> (B, 60, lstm_hidden_dim)
        
        # 3. 结合 LFS 输出和注意力上下文
        combined_features = torch.cat([lfs_output, attention_context], dim=-1)
        # -> (B, 60, lstm_hidden_dim * 2)

        # 4. 后融合处理与最终裁决
        final_sequence, (h_n, _) = self.post_fusion_processor(combined_features)
        
        # 使用序列的最后一个时间点的输出进行分类
        last_step_output = final_sequence[:, -1, :]
        logits = self.classifier(last_step_output)
        
        # 状态特征依然是最后一个LSTM的隐藏状态
        state_feature = h_n.squeeze(0) # -> (B, lstm_hidden_dim)

        return logits, state_feature
    

class ContextualFidelityDataset(Dataset):
    """
    Custom PyTorch Dataset to create sequences for the ContextualFidelityModel.
    Each sample consists of a sequence of features, a sequence of raw data,
    and the label corresponding to the last item in the sequence.
    """
    def __init__(self, features, raw_data, labels, sequence_length=4):
        self.features = features
        self.raw_data = raw_data
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        # The number of possible sequences is the total length minus the sequence length + 1
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        # The end index of the sequence slice
        end_idx = idx + self.sequence_length

        # Get the sequence of features and raw data
        feature_seq = self.features[idx:end_idx]
        raw_seq = self.raw_data[idx:end_idx]
        
        # The label corresponds to the final time step in the sequence
        label = self.labels[end_idx - 1]

        # Convert to tensors
        feature_seq_tensor = torch.tensor(feature_seq, dtype=torch.float32)
        raw_seq_tensor = torch.tensor(raw_seq, dtype=torch.float32)
        # Use float for labels for BCEWithLogitsLoss
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(-1)

        return feature_seq_tensor, raw_seq_tensor, label_tensor
    


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feature_seq, raw_seq, labels in loader:
            feature_seq, raw_seq, labels = feature_seq.to(device), raw_seq.to(device), labels.to(device)
            
            outputs, _ = model(feature_seq, raw_seq)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1





trial_arrays, trial_labels = load_data_from_structured_folders(DATASET_PATH)
SensorDataSequences, SensorLabelSequences = create_sequences(trial_arrays, trial_labels, WINDOW_SIZE, STEP)
print(f"The shape of the final dataset is: X={SensorDataSequences.shape}, y={SensorLabelSequences.shape}")


MODEL_PATH = "feature_model_1dcnn.pth"
SCALER_PATH = "scaler_50hz_torch.gz"

# 如果all_features.npy和all_labels.npy已经存在，且all_features.npy大小小于2GB，则直接加载并跳过后续处理
if os.path.exists("all_features.npy") and os.path.exists("all_labels.npy") and os.path.getsize("all_features.npy") < 2 * 1024**3:
    print(f"已加载现有的特征文件 'all_features.npy' 和标签文件 'all_labels.npy'，且大小符合要求。跳过后续处理。")
else:
    print("未找到现有的特征文件，开始生成新的特征文件...")
    # --- 加载模型和标准化器 ---
    print("正在加载模型和标准化器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = FeatureModel1DCNN(input_channels=11, num_classes=1).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"模型已从 {MODEL_PATH} 加载")
    else:
        print(f"警告: 在 {MODEL_PATH} 未找到模型文件。将使用随机初始化的模型。")
    model.eval() # 设置为评估模式

    # 加载标准化器
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"标准化器已从 {SCALER_PATH} 加载")
    else:
        # 抛出错误并停止执行
        raise FileNotFoundError(f"ERROR: Standard scaler file not found at '{SCALER_PATH}'. Cannot proceed without it.")


    # --- 批量处理数据并提取特征 ---
    print("\n开始批量提取特征...")
    all_features_list = []
    all_labels_list = []

    # `trial_arrays` 和 `trial_labels` 变量是从上一个数据加载单元格中获得的
    # 遍历每一次试验的数据
    for i, trial_data in enumerate(trial_arrays):
        trial_label = trial_labels[i]
        
        # 在当前试验数据上应用滑动窗口
        for j in range(0, len(trial_data) - WINDOW_SIZE + 1, STEP):
            # 1. 截取一个窗口的数据
            window_data = trial_data[j : j + WINDOW_SIZE]
            
            # 2. 预处理窗口数据 (标准化 -> 转换为Tensor)
            scaled_window = scaler.transform(window_data)
            window_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 3. 从模型中提取特征
            with torch.no_grad(): # 关闭梯度计算以加速
                features = model.extract_features(window_tensor)
            
            # 4. 将特征扁平化并添加到列表中
            flattened_features = features.cpu().numpy().flatten()
            all_features_list.append(flattened_features)
            
            # 5. 将该窗口对应的标签添加到列表中
            all_labels_list.append(trial_label)

    print(f"处理完成！共处理了 {len(trial_arrays)} 次试验，生成了 {len(all_features_list)} 个特征向量。")

    # --- 4. 保存最终的数据集 ---
    if all_features_list:
        # 将列表转换为Numpy数组
        final_features = np.array(all_features_list)
        final_labels = np.array(all_labels_list)

        # 保存数组到.npy文件
        np.save("all_features.npy", final_features)
        np.save("all_labels.npy", final_labels)

        print(f"\n数据集已成功保存:")
        print(f"  - 特征文件: all_features.npy, 形状: {final_features.shape}")
        print(f"  - 标签文件: all_labels.npy, 形状: {final_labels.shape}")
    else:
        print("\n未能生成任何特征，未创建文件。")


print("--- Starting Data Preparation ---")

# For demonstration, we'll create dummy files if they don't exist.
# In your case, these files should already be generated by your previous script.
if not os.path.exists("all_features.npy") or not os.path.exists("all_labels.npy"):
    print("File 'all_features.npy' or 'all_labels.npy' not found. Exiting.")
    raise FileNotFoundError("Required data files are missing.")


final_features = np.load("all_features.npy")
final_labels = np.load("all_labels.npy")
raw_windows_original = SensorDataSequences 

# 在这里定义稀疏度，例如 0.3 表示随机将 30% 的数据样本置零
SPARSITY_RATIO = 0.2 
raw_windows_sparse = create_sparse_data(raw_windows_original, SPARSITY_RATIO)

# 验证稀疏化是否成功 (可选)
# 一个有效样本的所有值求和后应大于0
# non_zero_samples_before = np.count_nonzero(np.sum(raw_windows_original, axis=(1, 2)))
# non_zero_samples_after = np.count_nonzero(np.sum(raw_windows_sparse, axis=(1, 2)))
# print(f"  - Non-zero samples before: {non_zero_samples_before}")
# print(f"  - Non-zero samples after: {non_zero_samples_after}\n")


# 定义常量
SEQUENCE_LENGTH = 4
BATCH_SIZE = 32
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# 创建 Dataset 实例时，使用稀疏化后的 `raw_windows_sparse`
full_dataset = ContextualFidelityDataset(final_features, raw_windows_sparse, final_labels, sequence_length=SEQUENCE_LENGTH)
print(f"Total number of sequences in the dataset: {len(full_dataset)}")

# Create indices for splitting
dataset_indices = list(range(len(full_dataset)))
train_val_indices, test_indices = train_test_split(dataset_indices, test_size=TEST_SIZE, random_state=42)
train_indices, val_indices = train_test_split(train_val_indices, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), random_state=42)

# Create subsets for train, validation, and test
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


print("\n--- Model Training and Evaluation ---")

# Model Hyperparameters
FEATURE_DIM = 6400  # from final_features.shape[1]
LSTM_HIDDEN_DIM = 256
# Calculate the output dimension of the raw_cnn
# Input: (B, 11, 200)
# After MaxPool1: 200 / 2 = 100
# After MaxPool2: 100 / 2 = 50
# After MaxPool3: 50 / 2 = 25
# Flattened output: 256 channels * 25 length = 6400
RAW_CNN_OUTPUT_DIM = 6400
NUM_CLASSES = 1
LEARNING_RATE = 0.0001
EPOCHS = 10

# Setup device, model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContextualFidelityModel(FEATURE_DIM, LSTM_HIDDEN_DIM, RAW_CNN_OUTPUT_DIM, NUM_CLASSES).to(device)
criterion = nn.BCEWithLogitsLoss() # Good for binary classification with one output neuron
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)




# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for i, (feature_seq, raw_seq, labels) in enumerate(train_loader):
        feature_seq, raw_seq, labels = feature_seq.to(device), raw_seq.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(feature_seq, raw_seq)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Print training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    
    # Evaluate on validation set
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, criterion, device)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

print("\nTraining finished.")

# 保存模型
torch.save(model.state_dict(), "contextual_fidelity_model.pth")

# --- 4. Final Evaluation on Test Set ---
print("\n--- Evaluating on Test Set ---")
test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

