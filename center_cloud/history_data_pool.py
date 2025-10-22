# history_data_pool.py

import numpy as np
from flask import Flask, jsonify, request
import threading
import joblib

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 5001
CONTROL_PORT = 5005  # 新增：控制端口
RAW_DATA_PATH = ".\\SensorDataSequences.npy"
SCALER_PATH = "autoregression_timeseries_data_scaler.save"
REQUEST_SAMPLE_COUNT = 4    # 发送 (x, 200, 11) 数据块时的样本数量，与模型训练时保持一致

# --- 全局变量 ---
SCALED_RAW_DATA = None
DATA_ITEM_SHAPE = None  # 用于存储单个数据项的形状，例如 (200, 11)
CURRENT_INDEX = 0
SEND_REAL_DATA = True   # 新增：控制标志，默认为发送真实数据
data_lock = threading.Lock() # 确保线程安全

def load_and_prepare_data():
    global SCALED_RAW_DATA, DATA_ITEM_SHAPE
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    try:
        raw_data = np.load(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: {RAW_DATA_PATH} not found. Exiting.")
        exit(1)
    
    if raw_data.ndim < 2:
        print(f"ERROR: Loaded data has insufficient dimensions: {raw_data.shape}. Exiting.")
        exit(1)

    print(f"Loaded raw data shape: {raw_data.shape}.")

    # 归一化处理
    print("\nApplying scaler to raw sensor data sequences...")
    scaler = joblib.load(SCALER_PATH)

    scaled_raw_data = []
    for i in range(raw_data.shape[0]):
        scaled_sequence = scaler.transform(raw_data[i])
        scaled_raw_data.append(scaled_sequence)

    SCALED_RAW_DATA = np.array(scaled_raw_data)

    print("Finished applying scaler to raw sensor data sequences.")
    
    # 存储单个数据项的形状 (例如: (200, 11))
    DATA_ITEM_SHAPE = raw_data.shape[1:]
    print(f"Data item shape set to: {DATA_ITEM_SHAPE}")
    print("History data pool is ready.")

# --- Flask 应用 1 (数据端口) ---
app_data = Flask(__name__)

@app_data.route('/get_raw_data_chunk', methods=['GET'])
def get_raw_data_chunk():
    global CURRENT_INDEX, SEND_REAL_DATA
    with data_lock:
        chunk_size = REQUEST_SAMPLE_COUNT
        if CURRENT_INDEX + chunk_size > len(SCALED_RAW_DATA):
            return jsonify({"error": "End of data"}), 404

        # 根据 SEND_REAL_DATA 标志决定发送什么
        if SEND_REAL_DATA:
            # 提取真实数据块
            data_chunk = SCALED_RAW_DATA[CURRENT_INDEX : CURRENT_INDEX + chunk_size]
            print(f"Serving REAL data chunk: indices {CURRENT_INDEX} to {CURRENT_INDEX + chunk_size - 1}")
        else:
            # 创建一个全零的数据块
            # 它的形状将是 (REQUEST_SAMPLE_COUNT, *DATA_ITEM_SHAPE)
            # 例如: (8, 200, 11)
            zero_shape = (chunk_size,) + DATA_ITEM_SHAPE
            data_chunk = np.zeros(zero_shape)
            print(f"Serving ZERO data chunk: indices {CURRENT_INDEX} to {CURRENT_INDEX + chunk_size - 1}")

        # 无论发送什么，索引都照常更新
        CURRENT_INDEX += chunk_size
        
        response = {
            "data_chunk": data_chunk.tolist()
        }
    return jsonify(response)

# --- Flask 应用 2 (控制端口) ---
app_control = Flask(__name__)

@app_control.route('/set_instruction', methods=['POST'])
def set_instruction():
    global SEND_REAL_DATA
    try:
        data = request.json
        if data is None or 'instruction' not in data:
            return jsonify({"error": "Missing 'instruction' in JSON body"}), 400
            
        instruction = data.get('instruction')

        with data_lock:
            if instruction == 1 or instruction is True:
                SEND_REAL_DATA = True
                message = "Instruction set to: SEND REAL DATA"
            elif instruction == 0 or instruction is False:
                SEND_REAL_DATA = False
                message = "Instruction set to: SEND ZERO DATA"
            else:
                return jsonify({"error": "Invalid instruction. Send 1 (true) or 0 (false)."}), 400
        
        print(message) # 在服务器日志中打印状态
        return jsonify({"status": "success", "message": message}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- 启动函数 ---
def run_data_server():
    print(f"Starting data server on http://{HOST}:{PORT}")
    app_data.run(host=HOST, port=PORT, debug=True, use_reloader=False)

def run_control_server():
    print(f"Starting control server on http://{HOST}:{CONTROL_PORT}")
    app_control.run(host=HOST, port=CONTROL_PORT, debug=True, use_reloader=False)

if __name__ == '__main__':
    load_and_prepare_data()

    # 在单独的线程中启动两个服务器
    data_thread = threading.Thread(target=run_data_server)
    control_thread = threading.Thread(target=run_control_server)

    data_thread.start()
    control_thread.start()

    data_thread.join()
    control_thread.join()