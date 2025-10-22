# model_runner_with_dqn.py

import requests
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
from collections import deque
from feature_model_definition import ContextualFidelityModel

import subprocess  # 用于启动子进程
import sys         # 用于获取python解释器路径
import atexit      # 作为一种备用关闭方案

# --- 配置 ---
HISTORY_DATA_POOL_URL = "http://127.0.0.1:5001/get_raw_data_chunk"
FEATURE_POOL_URL = "http://127.0.0.1:5002/get_feature"
INSTRUCTION_URL = "http://127.0.0.1:5005/set_instruction" # 新增: 控制DQN的动作
MODEL_PATH = ".\\contextual_fidelity_model_pretrained_encoder.pth"
REQUEST_INTERVAL_SECONDS = 0.05 # 每 x 秒请求一次特征
SEQUENCE_LENGTH = 4             # 累积 x 个 (REQUEST_SAMPLE_COUNT, 200, 11) 特征后进行一次推理

# --- DQN 超参数 ---
STATE_DIM = 256           # 状态维度 (来自 LSTM_HIDDEN_DIM)
ACTION_DIM = 2            # 动作维度 (0: 发送零向量, 1: 发送真实数据)
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99              # 折扣因子
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 10   # 每10步更新一次目标网络

# ===================================================================
# --- DQN 定义 ---
# ===================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.cat(next_states)
        
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state_tensor):
        """
        简化：以50%的概率输出1（同步/真实数据），50%的概率输出0（不同步/全零）
        """
        if random.random() > 0.5:
            action = 1 # 发送真实数据
        else:
            action = 0 # 发送全零向量
        return action

    def learn(self):
        """从经验回放池中采样并训练网络"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None # 缓冲区中数据不够

        states, actions, rewards, next_states = self.replay_buffer.sample(BATCH_SIZE)

        q_predicted = self.policy_net(states).gather(1, actions)
        q_next = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        q_target = rewards + (GAMMA * q_next)

        loss = self.loss_fn(q_predicted, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        """将 policy_net 的权重复制到 target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("\n*** Target network updated ***\n")


# --- HTTP 辅助函数 ---
def send_action(action):
    """向端口 5005 发送 'instruction' (动作 a)"""
    try:
        payload = {"instruction": int(action)} # 确保发送的是 0 或 1
        response = requests.post(INSTRUCTION_URL, json=payload)
        response.raise_for_status()
        print(f"  -> Action sent: {'SEND REAL DATA' if action == 1 else 'SEND ZERO DATA'}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending action: {e}")


# --- 保真模型加载 ---
def load_model(model_path):
    """加载训练好的模型并设置为评估模式"""
    print(f"Loading model from {model_path}...")
    
    # --- 模型超参数 (需要和训练时保持一致) ---
    LFS_FEATURE_DIM = 64
    HFS_FEATURE_DIM = 64 
    LSTM_HIDDEN_DIM = 256
    NUM_CLASSES = 1
    
    model = ContextualFidelityModel(
        lfs_feature_dim=LFS_FEATURE_DIM, 
        lstm_hidden_dim=LSTM_HIDDEN_DIM, 
        hfs_feature_dim=HFS_FEATURE_DIM, 
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    print("Model loaded successfully.")
    return model



# --- 主模拟/训练循环 ---
def simulate_training_loop(model, agent):
    """主模拟与DQN训练循环"""
    
    feature_sequence = []
    label_sequence = []
    
    total_predictions = 0
    correct_predictions = 0
    save_counter = 0
    param_rows = []

    # --- 获取初始状态 (s_0) ---
    print("Getting initial state (s_0)...")
    send_action(1) # 第一次强制发送真实数据
    
    # 第一次循环以获取 s_0
    try:
        for i in range(SEQUENCE_LENGTH):
            time.sleep(REQUEST_INTERVAL_SECONDS)
            response = requests.get(FEATURE_POOL_URL)
            response.raise_for_status()
            data = response.json()
            feature_sequence.append(data['feature'])
            label_sequence.append(data['label'])
        
        response = requests.get(HISTORY_DATA_POOL_URL)
        response.raise_for_status()
        raw_data_chunk = response.json()['data_chunk']
        
        feature_tensor = torch.tensor(np.array(feature_sequence), dtype=torch.float32).unsqueeze(0)
        raw_data_tensor = torch.tensor(np.array(raw_data_chunk), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            _, current_state = model(feature_tensor, raw_data_tensor) # (logits, state_features)
            
        feature_sequence.clear()
        label_sequence.clear()
        print("Initial state received. Starting training loop...")
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to get initial state: {e}. Exiting.")
        return
    # --- 初始状态获取完毕 ---

    step = 0
    while True:
        step += 1
        print("-" * 30)
        print(f"Step {step}")

        # 1. (DQN) 根据当前状态 s_t 选择动作 a_t
        action = agent.select_action(current_state)
        print(f"  - Selected Action: {action}")

        # 2. (HTTP) 执行动作 a_t (发送指示)
        send_action(action)
        
        # 3. (Runner) 运行环境一步：获取特征和原始数据
        try:
            for i in range(SEQUENCE_LENGTH):
                # print(f"Waiting {REQUEST_INTERVAL_SECONDS} seconds...") # 注释掉以加速
                time.sleep(REQUEST_INTERVAL_SECONDS)
                
                response = requests.get(FEATURE_POOL_URL)
                if response.status_code == 404:
                    print("Feature pool reports end of data. Simulation finished.")
                    return
                response.raise_for_status()
                
                data = response.json()
                feature_sequence.append(data['feature'])
                label_sequence.append(data['label'])

            response = requests.get(HISTORY_DATA_POOL_URL)
            if response.status_code == 404:
                print("History data pool reports end of data. Simulation finished.")
                return
            response.raise_for_status()
            
            raw_data_chunk = response.json()['data_chunk']
            raw_data_array = np.array(raw_data_chunk)

            # 4. (Runner) 准备模型输入
            feature_tensor = torch.tensor(np.array(feature_sequence), dtype=torch.float32).unsqueeze(0)
            raw_data_tensor = torch.tensor(raw_data_array, dtype=torch.float32).unsqueeze(0)
            
            # 5. (Runner) 观察 r_t 和 s_{t+1}
            print("  - Running RL model inference...")
            logits = None
            next_state = None
            
            # 只在获取奖励和下一个状态时使用 no_grad()
            # DQN 的 policy_net 训练是在 agent.learn() 中处理的
            with torch.no_grad():
                logits, next_state = model(feature_tensor, raw_data_tensor)
                
            prediction_prob = torch.sigmoid(logits).item()
            prediction = 1 if prediction_prob > 0.5 else 0
            
            # 6. (DQN) 定义奖励 (Reward Shaping)
            # 必须在存储到 buffer 之前定义
            true_label = label_sequence[-1]
            reward = 1.0 if prediction == true_label else -1.0
            
            # 7. (DQN) 存储 (s_t, a_t, r_t, s_{t+1})
            agent.replay_buffer.push(current_state, action, reward, next_state)
            
            # 8. (DQN) 更新 s_t = s_{t+1}
            current_state = next_state
            
            # 8. (DQN) 训练
            loss = agent.learn()
            if loss is not None:
                print(f"  - DQN Training Loss: {loss:.6f}")

            # 9. (DQN) 更新目标网络
            if step % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # 10. (Runner) 日志记录
           # 10. (Runner) 日志记录
            true_label = label_sequence[-1]
            total_predictions += 1
            is_correct = (prediction == true_label)
            if is_correct:
                correct_predictions += 1
            
            print(f"  - Prediction: {prediction} (Reward/Prob: {reward:.4f})")
            print(f"  - True Label: {true_label}")
            
            current_accuracy = (correct_predictions / total_predictions) * 100
            print(f"  - Cumulative Accuracy: {current_accuracy:.2f}% ({correct_predictions}/{total_predictions})")

            # 11. (Runner) 清空列表，为下一个序列做准备
            feature_sequence.clear()
            label_sequence.clear()

            # 12. (Runner) 统计预测情况
            zero_vectors_count = np.sum(np.all(raw_data_array == 0, axis=(1, 2)))
            param_rows.append({
                "Zero_Vectors_Ratio": zero_vectors_count / raw_data_array.shape[0],
                "Probability": reward,
                "Predict_Label": prediction,
                "True_Label": true_label,
                "Result": 1 if is_correct else 0,
                "DQN Action": action,
                "Cumulative_Accuracy": current_accuracy,
                "DQN Loss": loss if loss is not None else -1
            })

            if save_counter >= 10:
                df = pd.DataFrame(param_rows)
                df.to_csv("simulate_log_dqn.csv", mode='a', header=not pd.io.common.file_exists("simulate_log_dqn.csv"), index=False)
                print("Saved inference parameters to simulate_log_dqn.csv")
                save_counter = 0
                param_rows.clear()
            else:
                save_counter += 1

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred while communicating with a service: {e}")
            print("Will retry in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
            break


# --- 主程序入口 ---
# 分别启动版本
if __name__ == '__main__':
    # 1. 加载预训练的 ContextualFidelityModel
    context_model = load_model(MODEL_PATH)
    
    # 2. 初始化DQN Agent
    dqn_agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    # 3. 启动主训练循环
    simulate_training_loop(context_model, dqn_agent)


# 一键启动版本（有问题，暂时弃用）
# if __name__ == '__main__':
    
#     # --- 1. 定义子进程脚本路径 ---
#     FEATURE_POOL_SCRIPT = 'center_cloud\\feature_pool.py'
#     HISTORY_DATA_POOL_SCRIPT = 'center_cloud\\history_data_pool.py'
    
#     # 获取当前Python解释器的路径
#     # 这确保了子进程使用与主进程相同的Python环境
#     python_executable = sys.executable 
    
#     process_feature_pool = None
#     process_history_pool = None
    
#     try:
#         # --- 2. 启动子进程 ---
#         # Popen 是非阻塞的
#         print(f"Starting {FEATURE_POOL_SCRIPT} in background...")
#         process_feature_pool = subprocess.Popen(
#             [python_executable, FEATURE_POOL_SCRIPT],
#             stdout=subprocess.PIPE, # 捕捉标准输出
#             stderr=subprocess.PIPE  # 捕捉标准错误
#         )
        
#         print(f"Starting {HISTORY_DATA_POOL_SCRIPT} in background...")
#         process_history_pool = subprocess.Popen(
#             [python_executable, HISTORY_DATA_POOL_SCRIPT],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
        
#         # --- 3. 等待服务器启动 ---
#         print("Waiting 3 seconds for servers to initialize...")
#         time.sleep(3)
        
#         # 检查子进程是否在启动时就失败了
#         if process_feature_pool.poll() is not None:
#             raise RuntimeError(f"{FEATURE_POOL_SCRIPT} failed to start. Error:\n{process_feature_pool.stderr.read().decode()}")
        
#         if process_history_pool.poll() is not None:
#             raise RuntimeError(f"{HISTORY_DATA_POOL_SCRIPT} failed to start. Error:\n{process_history_pool.stderr.read().decode()}")

#         print("All servers started successfully.")

#         # --- 4. 运行主逻辑 ---
        
#         # 1. 加载预训练的 ContextualFidelityModel
#         context_model = load_model(MODEL_PATH)
        
#         # 2. 初始化DQN Agent
#         dqn_agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        
#         # 3. 启动主训练循环
#         simulate_training_loop(context_model, dqn_agent)
        
#     except KeyboardInterrupt:
#         print("\nMain loop interrupted by user (Ctrl+C). Shutting down...")
        
#     except Exception as e:
#         print(f"\nAn unexpected error occurred in the main loop: {e}")
        
#     finally:
#         # --- 5. 终止子进程 ---
#         print("\nShutting down subprocesses...")
#         if process_history_pool:
#             print(f"Terminating {HISTORY_DATA_POOL_SCRIPT} (PID: {process_history_pool.pid})...")
#             process_history_pool.terminate()
#             process_history_pool.wait()
#             print("History pool terminated.")

#         if process_feature_pool:
#             print(f"Terminating {FEATURE_POOL_SCRIPT} (PID: {process_feature_pool.pid})...")
#             process_feature_pool.terminate()
#             process_feature_pool.wait()
#             print("Feature pool terminated.")
            
#         print("All processes shut down.")


