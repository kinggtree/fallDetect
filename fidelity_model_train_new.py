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


DATASET_PATH = 'MobiFall_Dataset'
TARGET_SAMPLING_RATE_HZ = 50.0
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms"
WINDOW_SECONDS = 2
WINDOW_SIZE = int(TARGET_SAMPLING_RATE_HZ * WINDOW_SECONDS)

SEQUENCE_LENGTH = 8  # 每个序列包含x个时间步
STRIDE = 1           # 每隔x步创建一个新的序列

STEP_SECONDS = 1
STEP = int(TARGET_SAMPLING_RATE_HZ * STEP_SECONDS)

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
                df_resampled['acc_smv'] = np.sqrt(
                    df_resampled['acc_x']**2 + df_resampled['acc_y']**2 + df_resampled['acc_z']**2
                )

        elif sensor_code == 'gyro':
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

        resampled_dfs = {s_code: load_and_resample_sensor_file(sensor_files[s_code], s_code) for s_code in SENSOR_CODES}
        if any(df is None or df.empty for df in resampled_dfs.values()): continue

        try:
            common_start = max(df.index.min() for df in resampled_dfs.values())
            common_end = min(df.index.max() for df in resampled_dfs.values())
            if common_start >= common_end: continue

            aligned_dfs = [resampled_dfs[s_code][common_start:common_end].reset_index(drop=True) for s_code in SENSOR_CODES]
            if not all(len(df) > 0 and len(df) == len(aligned_dfs[0]) for df in aligned_dfs): continue
            
            combined_df = pd.concat(aligned_dfs, axis=1)
            
            if len(combined_df.columns) == len(ALL_FEATURE_COLUMNS):
                 combined_df.columns = ALL_FEATURE_COLUMNS
            else:
                 continue

            if len(combined_df) < WINDOW_SIZE: continue
            
            processed_trials_data.append(combined_df.values)
            labels.append(1 if trial_metadata_map[trial_key]["category"] == "FALLS" else 0)
            
        except Exception:
            continue

    print(f"Successfully processed and combined sensor data for {len(processed_trials_data)} trials.")
    return processed_trials_data, labels

def create_sequences(data_list, label_list, seq_length, step):
    X, y = [], []
    for i, trial_data in enumerate(data_list):
        trial_label = label_list[i]
        for j in range(0, len(trial_data) - seq_length + 1, step):
            X.append(trial_data[j:(j + seq_length)])
            y.append(trial_label)
            
    if not X: return np.array([]), np.array([])
    return np.array(X), np.array(y)

class FeatureModel1DCNN(nn.Module):
    def __init__(self, input_channels=11, num_classes=1, sequence_length=200): # 添加 sequence_length 参数
        super(FeatureModel1DCNN, self).__init__()
        
        # 特征提取器: 包含一系列的卷积和池化层
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2), # Length: L -> L/2
            
            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2), # Length: L/2 -> L/4

            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Length: L/4 -> L/8
        )
        
        # --- 动态计算分类器的输入维度 ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, sequence_length)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.numel()

        # 分类器: 将提取的特征映射到最终的输出
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512), # <-- 使用动态计算出的大小
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        features = self.feature_extractor(x)
        return features

def create_sparse_data(data_array, sparsity_ratio):
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError("Sparsity ratio must be between 0.0 and 1.0")
    sparse_array = data_array.copy()
    num_samples = sparse_array.shape[0]
    num_to_zero_out = int(num_samples * sparsity_ratio)
    if num_to_zero_out == 0:
        return sparse_array
    indices_to_zero = np.random.choice(
        np.arange(num_samples), 
        size=num_to_zero_out, 
        replace=False
    )
    sparse_array[indices_to_zero] = 0
    return sparse_array


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)

        x = x.permute(0, 1, 3, 2)
        x_reshape = x.contiguous().view(batch_size * time_steps, x.size(2), x.size(3))

        y = self.module(x_reshape)
        y = y.view(batch_size, time_steps, y.size(-1))
        
        return y
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim)
        self.key_layer = nn.Linear(key_dim, hidden_dim)
        self.value_layer = nn.Linear(key_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        context_vector = torch.matmul(attention_weights, V)
        return context_vector


def create_raw_data_cnn():
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

        raw_cnn = create_raw_data_cnn()
        self.hfs_processor = TimeDistributed(raw_cnn)

        self.lfs_processor = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.cross_attention = CrossAttention(
            query_dim=lstm_hidden_dim,
            key_dim=raw_cnn_output_dim,
            hidden_dim=lstm_hidden_dim
        )

        self.post_fusion_processor = nn.LSTM(
            input_size=lstm_hidden_dim * 2,
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
        lfs_output, _ = self.lfs_processor(feature_sequence)
        hfs_output = self.hfs_processor(imputed_raw_sequence)

        attention_context = self.cross_attention(
            query=lfs_output, 
            key=hfs_output, 
            value=hfs_output
        )
        combined_features = torch.cat([lfs_output, attention_context], dim=-1)

        final_sequence, (h_n, _) = self.post_fusion_processor(combined_features)
        
        last_step_output = final_sequence[:, -1, :]
        logits = self.classifier(last_step_output)

        state_feature = h_n.squeeze(0)

        return logits, state_feature
    

class ContextualFidelityDataset(Dataset):
    def __init__(self, features, raw_data, labels, sequence_length=4, stride=1):
        self.features = features
        self.raw_data = raw_data
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride
        self.num_sequences = (len(self.features) - self.sequence_length) // self.stride + 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        feature_seq = self.features[start_idx:end_idx]
        raw_seq = self.raw_data[start_idx:end_idx]
        label_slice = self.labels[start_idx:end_idx]
        label = np.max(label_slice)
        feature_seq_tensor = torch.tensor(feature_seq, dtype=torch.float32)
        raw_seq_tensor = torch.tensor(raw_seq, dtype=torch.float32)
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
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1




if __name__ == "__main__":
    SensorDataSequences, SensorLabelSequences = np.array([]), np.array([])
    trial_arrays, trial_labels = load_data_from_structured_folders(DATASET_PATH)

    if os.path.exists('SensorDataSequences.npy') and os.path.exists('SensorLabelSequences.npy'):
        print("Found existing npy files. Loading...")
        SensorDataSequences = np.load('SensorDataSequences.npy')
        print(f"Loaded dataset shape: X={SensorDataSequences.shape}")
        SensorLabelSequences = np.load('SensorLabelSequences.npy')
        print(f"Loaded dataset shape: y={SensorLabelSequences.shape}")
    else:
        SensorDataSequences, SensorLabelSequences = create_sequences(trial_arrays, trial_labels, WINDOW_SIZE, STEP)
        print(f"The shape of the final dataset is: X={SensorDataSequences.shape}, y={SensorLabelSequences.shape}")
        np.save('SensorDataSequences.npy', SensorDataSequences)
        np.save('SensorLabelSequences.npy', SensorLabelSequences)
        print("Saved processed dataset to npy files.")


    MODEL_PATH = "feature_model_1dcnn.pth"
    SCALER_PATH = "scaler_50hz_torch.gz"

    if os.path.exists("all_features.npy") and os.path.exists("all_labels.npy") and os.path.getsize("all_features.npy") < 2 * 1024**3:
        print(f"已加载现有的特征文件 'all_features.npy' 和标签文件 'all_labels.npy'，且大小符合要求。跳过后续处理。")
    else:
        print("未找到现有的特征文件，开始生成新的特征文件...")
        print("正在加载模型和标准化器...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        model = FeatureModel1DCNN(
            input_channels=11, 
            num_classes=1, 
            sequence_length=WINDOW_SIZE
        ).to(device)

        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"模型已从 {MODEL_PATH} 加载")
        else:
            raise FileNotFoundError(f"ERROR: Model file not found at '{MODEL_PATH}'. Cannot proceed without it.")
        model.eval()
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"标准化器已从 {SCALER_PATH} 加载")
        else:
            raise FileNotFoundError(f"ERROR: Standard scaler file not found at '{SCALER_PATH}'. Cannot proceed without it.")
        print("\n开始批量提取特征...")
        all_features_list = []
        all_labels_list = []
        for i, trial_data in enumerate(trial_arrays):
            trial_label = trial_labels[i]
            for j in range(0, len(trial_data) - WINDOW_SIZE + 1, STEP):
                window_data = trial_data[j : j + WINDOW_SIZE]
                scaled_window = scaler.transform(window_data)
                window_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.extract_features(window_tensor)
                flattened_features = features.cpu().numpy().flatten()
                all_features_list.append(flattened_features)
                all_labels_list.append(trial_label)
        print(f"处理完成！共处理了 {len(trial_arrays)} 次试验，生成了 {len(all_features_list)} 个特征向量。")
        if all_features_list:
            final_features = np.array(all_features_list)
            final_labels = np.array(all_labels_list)
            np.save("all_features.npy", final_features)
            np.save("all_labels.npy", final_labels)
            print(f"\n数据集已成功保存:")
            print(f"  - 特征文件: all_features.npy, 形状: {final_features.shape}")
            print(f"  - 标签文件: all_labels.npy, 形状: {final_labels.shape}")
        else:
            print("\n未能生成任何特征，未创建文件。")


    print("--- Starting Data Preparation ---")

    if not os.path.exists("all_features.npy") or not os.path.exists("all_labels.npy"):
        print("File 'all_features.npy' or 'all_labels.npy' not found. Exiting.")
        raise FileNotFoundError("Required data files are missing.")


    final_features = np.load("all_features.npy")
    final_labels = np.load("all_labels.npy")
    raw_windows_original = SensorDataSequences 

    SPARSITY_RATIO = 0.2 
    raw_windows = create_sparse_data(raw_windows_original, SPARSITY_RATIO)

    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.15

    print(f"Dataset Hyperparameters:")
    print(f"  - Sequence Length (time_steps): {SEQUENCE_LENGTH}")
    print(f"  - Stride: {STRIDE}")

    full_dataset = ContextualFidelityDataset(
        final_features, 
        raw_windows, 
        final_labels, 
        sequence_length=SEQUENCE_LENGTH, 
        stride=STRIDE
    )

    original_len = len(final_features) - SEQUENCE_LENGTH + 1
    new_len = len(full_dataset)
    print(f"\nDataset size changed:")
    print(f"  - Original size (stride=1): {original_len} sequences")
    print(f"  - New augmented size (stride={STRIDE}): {new_len} sequences")
    print(f"  - Augmentation factor: {new_len / original_len:.2f}x")

    full_dataset = ContextualFidelityDataset(final_features, raw_windows, final_labels, sequence_length=SEQUENCE_LENGTH)
    print(f"Total number of sequences in the dataset: {len(full_dataset)}")

    dataset_indices = list(range(len(full_dataset)))
    train_val_indices, test_indices = train_test_split(dataset_indices, test_size=TEST_SIZE, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), random_state=42)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    print("\n--- Model Training and Evaluation ---")

    # --- 动态计算模型超参数 ---
    # 使用一个虚拟张量来推断raw_cnn模块的输出维度
    dummy_cnn_input = torch.randn(1, 11, WINDOW_SIZE) 
    temp_cnn = create_raw_data_cnn()
    dummy_cnn_output = temp_cnn(dummy_cnn_input)
    RAW_CNN_OUTPUT_DIM = dummy_cnn_output.shape[1] # 获取展平后的特征维度

    # 特征维度也应该是动态的
    # final_features 是从 all_features.npy 加载的
    FEATURE_DIM = final_features.shape[1]

    # 验证两个维度是否一致 (理想情况下它们应该由同一个CNN结构产生)
    if FEATURE_DIM != RAW_CNN_OUTPUT_DIM:
        raise ValueError(f"Feature dimension from file ({FEATURE_DIM}) does not match calculated CNN output dimension ({RAW_CNN_OUTPUT_DIM}). Please check the feature extraction process.")

    LSTM_HIDDEN_DIM = 256
    NUM_CLASSES = 1
    LEARNING_RATE = 0.0001
    EPOCHS = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContextualFidelityModel(FEATURE_DIM, LSTM_HIDDEN_DIM, RAW_CNN_OUTPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss() # Good for binary classification with one output neuron
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (feature_seq, raw_seq, labels) in enumerate(train_loader):
            feature_seq, raw_seq, labels = feature_seq.to(device), raw_seq.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(feature_seq, raw_seq)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    print("\nTraining finished.")

    torch.save(model.state_dict(), "contextual_fidelity_model.pth")

    print("\n--- Evaluating on Test Set ---")
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

