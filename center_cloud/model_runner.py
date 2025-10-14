# model_runner.py

import requests
import numpy as np
import torch
import time
import torch.nn as nn


# --- 配置 ---
HISTORY_DATA_POOL_URL = "http://127.0.0.1:5001/get_raw_data_chunk"
FEATURE_POOL_URL = "http://127.0.0.1:5002/get_feature"
MODEL_PATH = ".\\contextual_fidelity_model.pth"
REQUEST_INTERVAL_SECONDS = 0.5 # 每x秒请求一次特征
SEQUENCE_LENGTH = 8            # 累积x个特征后进行一次推理

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.permute(0, 1, 3, 2) # -> (B, 60, 11, 200)
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
            hidden_dim=lstm_hidden_dim # 通常设置为与query_dim一致
        )
        
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
        lfs_output, _ = self.lfs_processor(feature_sequence) # -> (B, 60, lstm_hidden_dim)
        hfs_output = self.hfs_processor(imputed_raw_sequence) # -> (B, 60, raw_cnn_output_dim)
        attention_context = self.cross_attention(
            query=lfs_output, 
            key=hfs_output, 
            value=hfs_output
        )
        combined_features = torch.cat([lfs_output, attention_context], dim=-1)

        final_sequence, (h_n, _) = self.post_fusion_processor(combined_features)
        
        last_step_output = final_sequence[:, -1, :]
        logits = self.classifier(last_step_output)
        
        state_feature = h_n.squeeze(0) # -> (B, lstm_hidden_dim)

        return logits, state_feature






# --- 模型加载 ---
def load_model(model_path):
    """加载训练好的模型并设置为评估模式"""
    print(f"Loading model from {model_path}...")
    
    # --- 模型超参数 (需要和训练时保持一致) ---
    FEATURE_DIM = 3072
    LSTM_HIDDEN_DIM = 256
    RAW_CNN_OUTPUT_DIM = 3072
    NUM_CLASSES = 1
    
    model = ContextualFidelityModel(FEATURE_DIM, LSTM_HIDDEN_DIM, RAW_CNN_OUTPUT_DIM, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please train and save the model first.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    model.eval() # 设置为评估模式
    print("Model loaded successfully.")
    return model

def simulate_inference_loop(model):
    """主模拟循环"""
    feature_sequence = []
    label_sequence = []
    
    total_predictions = 0
    correct_predictions = 0

    while True:
        # 1. 累积一个序列长度的特征
        print("-" * 30)
        try:
            for i in range(SEQUENCE_LENGTH):
                print(f"Waiting {REQUEST_INTERVAL_SECONDS} seconds before requesting feature {i+1}/{SEQUENCE_LENGTH}...")
                time.sleep(REQUEST_INTERVAL_SECONDS)
                
                # 从特征池请求数据
                response = requests.get(FEATURE_POOL_URL)
                if response.status_code == 404:
                    print("Feature pool reports end of data. Simulation finished.")
                    return
                response.raise_for_status() # 如果发生其他HTTP错误则抛出异常
                
                data = response.json()
                feature_sequence.append(data['feature'])
                label_sequence.append(data['label'])
                print(f"-> Received feature {i+1}/{SEQUENCE_LENGTH}.")

            # 2. 从历史数据池请求对应的原始数据块
            print("\nRequesting raw data chunk from history pool...")
            response = requests.get(HISTORY_DATA_POOL_URL)
            if response.status_code == 404:
                print("History data pool reports end of data. Simulation finished.")
                return
            response.raise_for_status()
            
            raw_data_chunk = response.json()['data_chunk']
            print("-> Received raw data chunk.")
            raw_data_array = np.array(raw_data_chunk)


            zero_vectors_count = np.sum(np.all(raw_data_array == 0, axis=(1, 2)))
            print(f"全零向量占比: {zero_vectors_count / raw_data_array.shape[0]:.2f}")



            # 3. 准备模型输入
            feature_tensor = torch.tensor(np.array(feature_sequence), dtype=torch.float32).unsqueeze(0) # (1, 4, 6400)
            raw_data_tensor = torch.tensor(raw_data_array, dtype=torch.float32).unsqueeze(0) # (1, 4, 200, 11)
            
            # 4. 模型推理
            print("\nRunning model inference...")
            with torch.no_grad():
                logits, _ = model(feature_tensor, raw_data_tensor)
                
            prediction_prob = torch.sigmoid(logits).item()
            prediction = 1 if prediction_prob > 0.5 else 0
            
            # 5. 判断结果
            # 标签对应序列中的最后一个
            true_label = label_sequence[-1]
            
            total_predictions += 1
            is_correct = (prediction == true_label)
            if is_correct:
                correct_predictions += 1
            
            print(f"  - Prediction: {prediction} (Probability: {prediction_prob:.4f})")
            print(f"  - True Label: {true_label}")
            print(f"  - Result: {'CORRECT' if is_correct else 'WRONG'}")
            
            current_accuracy = (correct_predictions / total_predictions) * 100
            print(f"  - Cumulative Accuracy: {current_accuracy:.2f}% ({correct_predictions}/{total_predictions})")

            # 6. 清空列表，为下一个序列做准备
            feature_sequence.clear()
            label_sequence.clear()

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred while communicating with a service: {e}")
            print("Will retry in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
            break
            
if __name__ == '__main__':
    trained_model = load_model(MODEL_PATH)
    simulate_inference_loop(trained_model)