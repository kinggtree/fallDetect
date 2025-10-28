# feature_pool.py

import numpy as np
from flask import Flask, jsonify
import threading

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 5002
FEATURES_PATH = "all_features.npy"
LABELS_PATH = "all_labels.npy"

# --- 全局变量 ---
FEATURES = None
LABELS = None
CURRENT_INDEX = 0
data_lock = threading.Lock()
reset_timer = None # 用于跟踪重置定时器

def load_data():
    """加载特征和标签"""
    global FEATURES, LABELS
    print(f"Loading features from {FEATURES_PATH}...")
    try:
        FEATURES = np.load(FEATURES_PATH)
        LABELS = np.load(LABELS_PATH)
        print(f"Loaded features shape: {FEATURES.shape}, labels shape: {LABELS.shape}.")
    except FileNotFoundError:
        print("ERROR: Feature/Label files not found. Exiting.")
        exit(1)

    print("Feature pool is ready.")

# --- 重置索引的回调函数 ---
def reset_current_index():
    """因超时而重置 CURRENT_INDEX"""
    global CURRENT_INDEX
    with data_lock:
        # 检查索引是否已经为0，避免重复打印消息
        if CURRENT_INDEX != 0:
            CURRENT_INDEX = 0
            # 在打印前后添加换行符，使其在服务器日志中更显眼
            print("\n--- 5 seconds of inactivity. Resetting CURRENT_INDEX to 0. ---\n")


# ===================================================================
# --- 新增：用于 model_runner 直接调用的函数 ---
# ===================================================================
def get_feature_direct():
    """
    直接获取特征和标签，模拟 Flask 路由的逻辑。
    返回: (response_dict, status_code)
    """

    global CURRENT_INDEX
    with data_lock:
        
        if CURRENT_INDEX >= len(FEATURES):
            # 模拟 404 Not Found
            return ({"error": "End of data"}, 404)

        feature = FEATURES[CURRENT_INDEX]
        label = LABELS[CURRENT_INDEX]
        
        # print(f"Serving feature and label for index {CURRENT_INDEX}")

        # 更新索引
        CURRENT_INDEX += 1

        response = {
            "feature": feature.tolist(),
            "label": int(label) # 确保标签是标准的int类型
        }

    # 模拟 200 OK
    return (response, 200)




# --- Flask 应用 ---
app = Flask(__name__)

@app.route('/get_feature', methods=['GET'])
def get_feature():
    global CURRENT_INDEX, reset_timer
    with data_lock:
        # --- 取消上一个定时器 ---
        if reset_timer is not None:
            reset_timer.cancel()
        
        if CURRENT_INDEX >= len(FEATURES):
            return jsonify({"error": "End of data"}), 404

        feature = FEATURES[CURRENT_INDEX]
        label = LABELS[CURRENT_INDEX]
        
        print(f"Serving feature and label for index {CURRENT_INDEX}")

        # 更新索引
        CURRENT_INDEX += 1

        response = {
            "feature": feature.tolist(),
            "label": int(label) # 确保标签是标准的int类型
        }

        # --- 启动一个新的5秒定时器 ---
        reset_timer = threading.Timer(5.0, reset_current_index)
        reset_timer.start()
    return jsonify(response)

if __name__ == '__main__':
    load_data()
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)