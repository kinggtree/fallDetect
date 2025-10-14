# history_data_pool.py

import numpy as np
from flask import Flask, jsonify
import threading

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 5001
RAW_DATA_PATH = ".\\SensorDataSequences.npy"  # 假设您已将变量SensorDataSequences保存为SensorDataSequences.npy
SPARSITY_RATIO = 0.8     # 大幅稀疏化，例如80%的数据被置零

# --- 全局变量 ---
SPARSE_RAW_DATA = None
CURRENT_INDEX = 0
data_lock = threading.Lock() # 确保线程安全

def create_sparse_data(data_array, sparsity_ratio):
    print("Sparsifying data...")
    sparse_array = data_array.copy()
    num_samples = sparse_array.shape[0]
    num_to_zero_out = int(num_samples * sparsity_ratio)
    indices_to_zero = np.random.choice(np.arange(num_samples), size=num_to_zero_out, replace=False)
    sparse_array[indices_to_zero] = 0
    print(f"Sparsification complete. {num_to_zero_out}/{num_samples} samples zeroed.")
    return sparse_array

def load_and_prepare_data():
    """加载并稀疏化原始数据"""
    global SPARSE_RAW_DATA
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    # 为了能独立运行，如果文件不存在则创建一个虚拟的X
    try:
        raw_data = np.load(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: {RAW_DATA_PATH} not found. Exiting.")
        exit(1)

    SPARSE_RAW_DATA = create_sparse_data(raw_data, SPARSITY_RATIO)
    print("History data pool is ready.")

# --- Flask 应用 ---
app = Flask(__name__)

@app.route('/get_raw_data_chunk', methods=['GET'])
def get_raw_data_chunk():
    global CURRENT_INDEX
    with data_lock:
        chunk_size = 4 # 客户端每次请求4个样本
        if CURRENT_INDEX + chunk_size > len(SPARSE_RAW_DATA):
            return jsonify({"error": "End of data"}), 404

        # 提取数据块
        data_chunk = SPARSE_RAW_DATA[CURRENT_INDEX : CURRENT_INDEX + chunk_size]
        
        print(f"Serving raw data chunk: indices {CURRENT_INDEX} to {CURRENT_INDEX + chunk_size - 1}")

        # 更新索引
        CURRENT_INDEX += chunk_size
        
        response = {
            "data_chunk": data_chunk.tolist()
        }
    return jsonify(response)

if __name__ == '__main__':
    load_and_prepare_data()
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)