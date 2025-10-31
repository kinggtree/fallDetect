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

import os
from torch.utils.tensorboard import SummaryWriter

# --- (新增) 导入本地模块的函数 ---
from pools_function import load_and_prepare_data as load_history_data, get_raw_data_slice_direct, set_instruction_direct, get_feature_direct, next_subject_direct
# ---------------------------------

timestamp = time.strftime("%Y%m%d_%H%M%S")

# --- (移除) 网络配置 ---
# HISTORY_DATA_POOL_URL = "http://127.0.0.1:5001/get_raw_data_chunk"
# FEATURE_POOL_URL = "http://127.0.0.1:5002/get_feature"
# INSTRUCTION_URL = "http://127.0.0.1:5005/set_instruction" # 控制DQN的动作
# ---------------------------------
MODEL_PATH = ".\\contextual_fidelity_model_pretrained_encoder.pth"
LOG_PATH = f"model_runner_dqn_log_{timestamp}.csv"
REQUEST_INTERVAL_SECONDS = 0.05 # 每 x 秒请求一次特征
SEQUENCE_LENGTH = 4             # 累积 x 个 (REQUEST_SAMPLE_COUNT, 200, 11) 特征后进行一次推理

# --- DQN 超参数 ---
STATE_DIM = 256           # 状态维度 (来自保真模型 LSTM_HIDDEN_DIM)
ACTION_DIM = 2            # 动作维度 (0: 发送零向量, 1: 发送真实数据)
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99              # 折扣因子
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 200   # 每 x 步更新一次目标网络 (不必太频繁地更新目标网络)
COST_PENALTY = 0.1        # 新增：每次选择 action=1 (同步) 时的惩罚值

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

    def push(self, state, action, reward, next_state, done):  # episode 终止时, done = 1, 否则 done = 0
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.int64).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, is_test: bool = False, is_tb: bool = False):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.loss_fn = nn.SmoothL1Loss()
        
        # training or testing
        self.is_test = is_test
        # epsilon-greedy related hyper-params
        self.epsilon = 1.0
        self.epsilon_decay_cnt = 10000  # number of updates required to decay to the minimum
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        # log related
        self.update_cnt = 0
        self.writer_wnd = 100
        self.train_fidelities, self.train_rewards = deque(maxlen=self.writer_wnd), deque(maxlen=self.writer_wnd)
        self.is_tb = is_tb
        tb_path = f"logs/tb/run_{timestamp}"
        if is_tb:
            os.makedirs(tb_path, exist_ok=True)
            self.writer = SummaryWriter(tb_path)

    def select_action(self, state_tensor):
        """
        简化：以50%的概率输出1（同步/真实数据），50%的概率输出0（不同步/全零）
        """
        # if random.random() > 0.5:
        #     action = 1 # 发送真实数据
        # else:
        #     action = 0 # 发送全零向量
        # return action

        if self.is_test:  # greedy selection
            action = self.policy_net(state_tensor).argmax().detach().item()
            return action
        else:  # epsilon-greedy selection
            if self.epsilon > np.random.random():
                action = random.randint(0, ACTION_DIM - 1)
            else:
                action = self.policy_net(state_tensor).argmax().detach().item()
            return action

    def learn(self):
        """从经验回放池中采样并训练网络"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None # 缓冲区中数据不够

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        q_predicted = self.policy_net(states).gather(1, actions)
        q_next = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        q_target = rewards + (1 - dones) * GAMMA * q_next  # 终止状态时不再考虑未来的 Q 值

        loss = self.loss_fn(q_predicted, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1

        if self.update_cnt % 100 == 0:
            r = sum(self.train_rewards) / len(self.train_rewards)
            f = sum(self.train_fidelities) / len(self.train_fidelities)
            # print("R {:4f} || F {:4f}".format(r, f))
            if self.is_tb:
                self.writer.add_scalar("measure/reward", r, self.update_cnt)
                self.writer.add_scalar("measure/accuracy", f, self.update_cnt)
                # self.writer.add_scalar("measure/epsilon", self.epsilon, self.update_cnt)
                # self.writer.add_scalar("measure/loss", loss, self.update_cnt)

        # linearly decrease epsilon
        self.epsilon = max(
            self.min_epsilon, 
            self.epsilon - (self.max_epsilon - self.min_epsilon) / self.epsilon_decay_cnt
        )

        # update the target_net
        if self.update_cnt % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
        
        return loss.item()

    def update_target_network(self):
        """将 policy_net 的权重复制到 target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("\n*** Target network updated ***\n")

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

    def load_model(self, filename):
        ckp_states = torch.load(filename)
        self.policy_net.load_state_dict(ckp_states["model_state_dict"])
        self.optimizer.load_state_dict(ckp_states["optimizer_state_dict"])


# --- (修改) HTTP 辅助函数 ---
def send_action(action):
    """(修改) 直接调用函数发送 'instruction' (动作 a)"""
    try:
        response, status_code = set_instruction_direct(int(action))
        if status_code != 200:
            print(f"Error sending action: {response.get('error', 'Unknown error')}")
        # 成功消息已在 set_instruction_direct 内部打印
    except Exception as e:
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
def simulate_training_loop(model, agent: DQNAgent):
    """主模拟与DQN训练循环"""

    is_print = False
    
    feature_sequence = []
    label_sequence = []
    
    total_predictions = 0
    correct_predictions = 0
    save_counter = 0
    param_rows = []

    actions_list = []

    history_data_sequence = []

    position_index = 4

    is_done = 0

    

    # --- 获取初始状态 (s_0) ---
    print("Getting initial state (s_0)...")

    
    # 第一次循环以获取 s_0
    for _ in range(SEQUENCE_LENGTH):
        send_action(1) # 第一批强制发送真实数据

        actions_list.append(1)

        raw_data_response, raw_status_code = get_raw_data_slice_direct()
        if raw_status_code != 200:
                raise Exception(f"Failed to get history slice: {raw_data_response.get('error')}")
        history_data_sequence.append(raw_data_response['data_slice'])
        
        data, status_code = get_feature_direct()
        if status_code != 200:
            raise Exception(f"Failed to get feature: {data.get('error')}")
        
        feature_sequence.append(data['feature'])
        label_sequence.append(data['label'])
        
        # 检查初始受试者是否在初始化期间就结束了
        is_done = raw_data_response['is_done']
        
    
    
    feature_tensor = torch.tensor(np.array(feature_sequence[position_index-4:position_index]), dtype=torch.float32).unsqueeze(0)
    raw_data_tensor = torch.tensor(np.array(history_data_sequence[position_index-4:position_index]), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        _, current_state = model(feature_tensor, raw_data_tensor) # (logits, state_features)
        
    position_index += 1
    print("Initial state received. Starting training loop...")
        
    # --- 初始状态获取完毕 ---

    step = 0
    while True:
        step += 1
        if is_print:
            print(f"Current Step: {step}")

        # 1. (DQN) 根据当前状态 s_t 选择动作 a_t
        action = agent.select_action(current_state)
        actions_list.append(action)

        # 2. 执行动作 a_t (发送指示)
        send_action(action)
        
        # 3. (Runner) 运行环境一步：获取特征和原始数据

        # 将一次获取4条历史数据改为一次获取1条
        history_data_response, history_status_code = get_raw_data_slice_direct()
        # 【修改】不再检查404，只检查非200的通用错误
        if history_status_code != 200:
            raise Exception(f"Error from history pool: {history_data_response.get('error')}")
            
        # 直接调用函数
        data, status_code = get_feature_direct()
        # 【修改】不再检查404
        if status_code != 200:
            raise Exception(f"Error from feature pool: {data.get('error')}")
        
        feature_sequence.append(data['feature'])
        label_sequence.append(data['label'])
        history_data_sequence.append(history_data_response['data_slice'])

        # 【修改】将 'is_done' 捕获到一个局部变量 'is_done_flag'
        is_done_flag = history_data_response['is_done']

        # 4. (Runner) 准备模型输入
        feature_tensor = torch.tensor(np.array(feature_sequence[position_index-4:position_index]), dtype=torch.float32).unsqueeze(0)
        raw_data_tensor = torch.tensor(np.array(history_data_sequence[position_index-4:position_index]), dtype=torch.float32).unsqueeze(0)

        # 5. (Fidelity Model Runner) 观察 r_t 和 s_{t+1}
        logits = None
        next_state = None
        
        with torch.no_grad():
            logits, next_state = model(feature_tensor, raw_data_tensor)
            
        prediction_prob = torch.sigmoid(logits).item()
        prediction = 1 if prediction_prob > 0.5 else 0
        
       # 6. (DQN) 定义奖励 (Reward Shaping)
        true_label = label_sequence[-1]
        
        if true_label == 1:
            accuracy_reward = 2 * prediction_prob - 1
        else: 
            accuracy_reward = 1 - 2 * prediction_prob
        
        action_cost = 0.0
        if action == 1:
            action_cost = COST_PENALTY 
        
        reward = accuracy_reward - action_cost

        agent.train_rewards.append(reward)
        if prediction == true_label:
            agent.train_fidelities.append(1)
        else:
            agent.train_fidelities.append(0)
        
        # 7. (DQN) 存储 (s_t, a_t, r_t, s_{t+1}, done)
        # 【修改】使用 is_done_flag 来确定 'done' (1 或 0)
        done = 1 if is_done_flag else 0
        agent.replay_buffer.push(current_state, action, reward, next_state, done)
        if is_done_flag:
            print("Done once.")
        
        # 8. (DQN) 更新 s_t = s_{t+1}
        current_state = next_state
        
        # 8. (DQN) 训练
        loss = agent.learn()

        # 10. (Runner) 日志记录
        true_label = label_sequence[-1]
        total_predictions += 1
        is_correct = (prediction == true_label)
        if is_correct:
            correct_predictions += 1
            
        current_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        # 12. (Runner) 统计过去 100 次预测情况
        statistic_length = 100
        if save_counter >= statistic_length:
            current_seq = np.array(history_data_sequence[position_index-statistic_length:position_index])
            zero_vectors_count = np.sum(np.all(current_seq == 0, axis=(1, 2)))
            action1_temp_list = actions_list[position_index-statistic_length:position_index]
            action1_ratio = sum(a == 1 for a in action1_temp_list) / len(action1_temp_list) if action1_temp_list else 0
            param_rows.append({
                "Current_Step": position_index,
                "DQN Loss": loss if loss is not None else -1,
                "Zero_Vectors_Ratio": '{:.2f}'.format(zero_vectors_count / statistic_length),
                "Cumulative_Accuracy": current_accuracy,
                "Action_1_Ratio": '{:.2f}'.format(action1_ratio)
            })
            df = pd.DataFrame(param_rows)
            df.to_csv(LOG_PATH, mode='a', header=not pd.io.common.file_exists(LOG_PATH), index=False)
            save_counter = 0
            param_rows.clear()
            is_print = True
            correct_predictions = 0
            total_predictions = 0
        else:
            save_counter += 1
            is_print = False
        
        # 13. (Runner) 更新 position_index
        position_index += 1

        # --- 14. 处理 Episode 结束 ---
        # 我们使用 'while' 循环来处理边缘情况：即新受试者的数据也可能非常短（< 4）
        while is_done_flag:
            print(f"\n--- Subject finished (Step {step}). Moving to next subject... ---")

            # 14a. 告诉数据池切换到下一个受试者
            response, status_code = next_subject_direct()

            if status_code == 404:
                # 这是真正的训练结束，所有受试者都已处理完毕
                print("All subjects processed. Simulation finished.")
                return  # 退出 simulate_training_loop

            # 14b. 为新受试者获取新的初始状态 s_0
            print("Getting initial state (s_0) for new subject...")
            is_done_flag = False  # 重置标志，假设新受试者是正常的
            
            for i in range(SEQUENCE_LENGTH):
                send_action(1)  # 强制发送真实数据
                actions_list.append(1)

                raw_data_response, raw_status_code = get_raw_data_slice_direct()
                if raw_status_code != 200:
                    print(f"Error getting initial data for new subject: {raw_data_response.get('error')}. Stopping.")
                    return
                
                # 检查这个新受试者是否在初始化时也结束了
                if raw_data_response['is_done']:
                    is_done_flag = True # 这将使 'while is_done_flag' 循环再次运行

                history_data_sequence.append(raw_data_response['data_slice'])

                data, status_code = get_feature_direct()
                if status_code != 200:
                    print(f"Error getting initial feature for new subject: {data.get('error')}. Stopping.")
                    return

                feature_sequence.append(data['feature'])
                label_sequence.append(data['label'])

                # 为这些初始化步骤更新计数器
                position_index += 1
                step += 1

            # 14c. 计算新的 s_0
            feature_tensor_init = torch.tensor(np.array(feature_sequence[position_index-4:position_index]), dtype=torch.float32).unsqueeze(0)
            raw_data_tensor_init = torch.tensor(np.array(history_data_sequence[position_index-4:position_index]), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                # 用新的 s_0 覆盖 'current_state'
                _, current_state = model(feature_tensor_init, raw_data_tensor_init)

            if is_done_flag:
                print("--- WARNING: New subject finished during initialization. Looping to next subject... ---")
            else:
                print("New initial state received. Continuing training...")
                # 如果 'is_done_flag' 为 False, 'while' 循环将在下次检查时退出


# --- 主程序入口 ---    
if __name__ == '__main__':
    
    # 0. (新增) 首先加载所有数据
    print("--- Initializing Data Pools ---")
    load_history_data()
    print("--- Data Pools Ready ---")

    # 1. 加载预训练的 ContextualFidelityModel
    context_model = load_model(MODEL_PATH)
    
    # 2. 初始化DQN Agent
    dqn_agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, is_test=False, is_tb=True)
    
    # 3. 启动主训练循环
    simulate_training_loop(context_model, dqn_agent)

    # 4. 训练结束后保存模型
    # (无论是因为数据耗尽还是Ctrl+C，循环结束后都会执行到这里)
    print("\nTraining loop finished. Saving DQN agent model...")
    save_path = "dqn_agent_final.pth"
    dqn_agent.save_model(save_path)
    print(f"DQN model saved to {save_path}")

