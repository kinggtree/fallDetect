import torch
import torch.nn as nn
import numpy as np
import os
import glob
import joblib

# 模型参数
INPUT_DIM = 11
HIDDEN_DIM = 64
N_LAYERS = 2
MODEL_PATH = 'autoregression_feature_extractor_model.pt'
DROP_OUT = 0.1

# --- 1. 定义路径 ---

# 包含按受试者处理好的 .npy 文件的目录
SUBJECT_DATA_DIR = 'Processed_Per_Subject' 

# 你的缩放器路径
SCALER_PATH = 'autoregression_timeseries_data_scaler.save'

# 【新】输出目录：用于存放按受试者保存的特征文件
OUTPUT_FEATURE_DIR = 'Extracted_Features_Per_Subject'

# 创建输出目录（如果不存在）
if not os.path.exists(OUTPUT_FEATURE_DIR):
    os.makedirs(OUTPUT_FEATURE_DIR)
    print(f"Created output directory: {OUTPUT_FEATURE_DIR}")



# --- 2. 初始化特征提取器 ---
print("\n--- Initializing Feature Extractor ---")
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell
        

# --- 核心：特征提取器类 ---
class FeatureExtractor:
    def __init__(self, model_path, scaler_path, input_dim, hidden_dim, n_layers, dropout=0.0):
        """
        初始化特征提取器。
        """
        # 检查设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FeatureExtractor is using device: {self.device}")

        # 1. 加载训练好的 Scaler
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Successfully loaded scaler from {scaler_path}")
        except FileNotFoundError:
            print(f"Error: Scaler file not found at {scaler_path}")
            raise

        # 1. 实例化我们需要的Encoder模型
        self.encoder = Encoder(input_dim, hidden_dim, n_layers, dropout).to(self.device)

        # 2. 加载训练好的完整Seq2Seq模型的权重字典
        full_state_dict = torch.load(model_path, map_location=self.device)

        # 3. 创建一个新的字典，只包含Encoder的权重
        #    并移除键名前缀 "encoder."
        encoder_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('encoder.'):
                # 将 'encoder.lstm.weight_ih_l0' 变为 'lstm.weight_ih_l0'
                new_key = key[len('encoder.'):] 
                encoder_state_dict[new_key] = value
        
        # 4. 将筛选后的权重加载到Encoder模型中
        self.encoder.load_state_dict(encoder_state_dict)
        
        print(f"Successfully loaded encoder weights from {model_path}")

        # 5. 设置为评估模式
        self.encoder.eval()

    def extract_feature(self, sequence_data):
        """
        从一个4秒(200个点)的序列中提取特征向量。

        参数:
            sequence_data (np.ndarray): 输入的传感器数据，形状必须为 (200, 11)

        返回:
            np.ndarray: 提取出的特征向量，形状为 (hidden_dim,)
        """
        # --- 输入验证 ---
        if not isinstance(sequence_data, np.ndarray) or sequence_data.shape != (200, 11):
            raise ValueError("Input data must be a numpy array of shape (200, 11)")

        # --- 特征提取核心逻辑 ---
        with torch.no_grad(): # 关闭梯度计算，加速推理
            # 0. 使用加载的 scaler 对【原始输入数据】进行归一化
            # scaler.transform期望一个2D数组，输入(200, 11)正好符合
            scaled_sequence_data = self.scaler.transform(sequence_data)
            
            # 1. 将【归一化后】的数据转换为PyTorch张量
            input_tensor = torch.tensor(scaled_sequence_data, dtype=torch.float32).to(self.device)

            # 2. 增加Batch维度
            # 模型的LSTM层期望的输入是 (batch_size, seq_len, input_dim)
            # 所以 (200, 11) 需要变成 (1, 200, 11)
            input_tensor = input_tensor.unsqueeze(0)

            # 3. 通过Encoder进行前向传播
            hidden_state, _ = self.encoder(input_tensor)
            # hidden_state 的形状是 (n_layers, batch_size, hidden_dim)

            # 4. 提取我们需要的特征向量
            # 通常我们使用最后一层的隐藏状态作为特征
            feature_vector_tensor = hidden_state[-1, :, :] # 取最后一层, shape: (1, hidden_dim)

            # 5. 去掉Batch维度，并转换回Numpy数组
            feature_vector_tensor = feature_vector_tensor.squeeze(0) # Shape: (hidden_dim)
            feature_vector_np = feature_vector_tensor.cpu().numpy()

            return feature_vector_np

# --- 2. 初始化特征提取器 ---
print("\n--- Initializing Feature Extractor ---")
extractor = FeatureExtractor(
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    dropout=DROP_OUT
)

# --- 3. 遍历所有受试者文件，提取特征并单独保存 ---
print(f"\n--- Loading data from '{SUBJECT_DATA_DIR}' ---")

# 查找所有按受试者保存的数据和标签文件
subject_data_files = sorted(glob.glob(os.path.join(SUBJECT_DATA_DIR, 'SensorDataSequences_sub*.npy')))

# 验证文件
if not subject_data_files:
    print(f"错误：在 '{SUBJECT_DATA_DIR}' 中找不到 'SensorDataSequences_sub*.npy' 文件。")
    exit(1)

print(f"找到了 {len(subject_data_files)} 个受试者的数据文件。")

total_sequences_processed = 0

# 遍历每一个受试者文件
for data_path in subject_data_files:
    base_data_name = os.path.basename(data_path)
    print(f"\nProcessing: {base_data_name}...")
    
    # 加载单个受试者的数据和标签
    subject_sequences = np.load(data_path)
    
    num_sequences = subject_sequences.shape[0]
    
    print(f"  Loaded {num_sequences} sequences.")

    # 为该受试者提取特征
    subject_features_temp_list = []
    for i in range(num_sequences):
        sequence = subject_sequences[i] # 取出第 i 个序列
        feature = extractor.extract_feature(sequence) # 提取特征
        subject_features_temp_list.append(feature) # 添加到临时列表中
        
        # 打印进度
        if (i + 1) % 500 == 0 or (i + 1) == num_sequences:
            print(f"  ... processed {i + 1}/{num_sequences} sequences for this subject.")

    # 将该受试者的所有特征转换为一个Numpy数组
    subject_features_np = np.array(subject_features_temp_list)
    print(f"  > Subject features shape: {subject_features_np.shape}")
    
    # --- 4. 【新】保存该受试者的特征和标签 ---
    
    # 从原始文件名 'SensorDataSequences_sub1.npy' 提取 '1'
    subject_id_str = base_data_name.replace('SensorDataSequences_sub', '').replace('.npy', '')
    
    # 构建新的输出文件名
    output_feature_path = os.path.join(OUTPUT_FEATURE_DIR, f'Features_sub{subject_id_str}.npy')
    
    # 保存文件
    np.save(output_feature_path, subject_features_np)
    
    print(f"  > Saved features to: {output_feature_path}")
    
    total_sequences_processed += num_sequences

# --- 5. 结束 ---
print(f"\n--- 5. Processing Complete ---")
print(f"所有受试者的特征文件已单独保存到 '{OUTPUT_FEATURE_DIR}' 目录。")
print(f"总共处理了 {total_sequences_processed} 个序列。")