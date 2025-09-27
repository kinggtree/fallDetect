import socket
import struct
import pickle
import numpy as np
import os
import torch
import joblib
from collections import deque

# 导入共享配置和模型定义
from shared_config import HOST, PORT
class FeatureModel1DCNN(torch.nn.Module):
    def __init__(self, input_channels=11, num_classes=1):
        super(FeatureModel1DCNN, self).__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding='same'), torch.nn.ReLU(), torch.nn.BatchNorm1d(64),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding='same'), torch.nn.ReLU(), torch.nn.BatchNorm1d(128),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same'), torch.nn.ReLU(), torch.nn.BatchNorm1d(256),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 25, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
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


# --- 预留的网络接口，用于将特征发送到保真模型 ---
def send_feature_to_server(feature_vector: np.ndarray):
    """
    将提取的特征向量发送到中央服务器（保真模型）。
    这是一个预留的接口。
    
    :param feature_vector: 从模型中提取的特征，已展平为一维数组。
    """
    # ==================== FUTURE IMPLEMENTATION HERE ====================
    # 在未来，这里可以实现:
    # 1. 使用 requests 库发送一个 HTTP POST 请求到云端API。
    #    requests.post("http://your-server.com/api/features", json=feature_vector.tolist())
    # 2. 使用另一个 TCP socket 连接到保真模型服务。
    # 3. 使用消息队列 (如 ZeroMQ, RabbitMQ) 发送消息。
    #
    # 目前，我们只打印信息来模拟这个动作。
    # ====================================================================
    # print(f"  [Network Interface] --> Sent feature of shape {feature_vector.shape} to server.")
    pass # 暂时禁用打印，避免刷屏

class EdgeInferenceEngine:
    """封装边缘推理的所有逻辑。"""
    def __init__(self, model_path, scaler_path, device):
        # ... (这部分逻辑和之前的 EdgeNodeSimulator 基本相同) ...
        self.WINDOW_SIZE = 200
        self.STEP_SIZE = 25 # 0.5秒
        self.OUTPUT_DIR = "simulated_features_live"
        
        self.model = FeatureModel1DCNN(input_channels=11).to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None # 应该要报错，但为了模拟先这样
            
        self.device = device
        self.buffer = deque(maxlen=self.WINDOW_SIZE)
        self.points_since_last_inference = 0
        if not os.path.exists(self.OUTPUT_DIR): os.makedirs(self.OUTPUT_DIR)
        
    def process_data_point(self, data_point, timestamp):
        self.buffer.append(data_point)
        self.points_since_last_inference += 1
        
        if len(self.buffer) == self.WINDOW_SIZE and self.points_since_last_inference >= self.STEP_SIZE:
            self.points_since_last_inference = 0
            
            current_window = np.array(self.buffer)
            scaled_window = self.scaler.transform(current_window)
            window_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.extract_features(window_tensor)
                logits = self.model(window_tensor)
            
            confidence = torch.sigmoid(logits).item()
            prediction = "FALL DETECTED!" if confidence > 0.5 else "No Fall"
            
            print(f"Timestamp: {timestamp:7.2f}s | Confidence: {confidence:.4f} | Prediction: {prediction}")
            
            # 调用预留的网络接口发送特征
            feature_flat = features.cpu().numpy().flatten()
            send_feature_to_server(feature_flat)
            # 可以在这里也保存特征到本地文件，作为备份
            # ...

def receive_all(sock, n):
    """辅助函数，确保从socket接收到指定长度的数据。"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = EdgeInferenceEngine("feature_model_1dcnn.pth", "scaler_50hz_torch.gz", device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Edge Inference Node is listening on {HOST}:{PORT}...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            total_points = 0
            header_size = struct.calcsize('!I')
            
            while True:
                # 1. 读取消息头
                header_bytes = receive_all(conn, header_size)
                if not header_bytes:
                    break # 连接已关闭
                
                # 2. 解析消息长度
                msg_len = struct.unpack('!I', header_bytes)[0]
                
                # 3. 读取完整消息体
                data_bytes = receive_all(conn, msg_len)
                if not data_bytes:
                    break
                    
                # 4. 反序列化数据
                data_point = pickle.loads(data_bytes)
                
                # 5. 处理数据点
                timestamp = total_points / 50.0
                engine.process_data_point(data_point, timestamp)
                total_points += 1

    print("Connection closed. Inference node shutting down.")

if __name__ == '__main__':
    main()