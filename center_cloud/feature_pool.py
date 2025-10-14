# feature_pool.py

import numpy as np
from flask import Flask, jsonify
import threading

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 5002
FEATURES_PATH = ".\\all_features.npy"
LABELS_PATH = ".\\all_labels.npy"

# --- 全局变量 ---
FEATURES = None
LABELS = None
CURRENT_INDEX = 0
data_lock = threading.Lock()

def load_data():
    """加载特征和标签"""
    global FEATURES, LABELS
    print(f"Loading features from {FEATURES_PATH}...")
    try:
        FEATURES = np.load(FEATURES_PATH)
        LABELS = np.load(LABELS_PATH)
    except FileNotFoundError:
        print("ERROR: Feature/Label files not found. Exiting.")
        exit(1)

    print("Feature pool is ready.")


# --- Flask 应用 ---
app = Flask(__name__)

@app.route('/get_feature', methods=['GET'])
def get_feature():
    global CURRENT_INDEX
    with data_lock:
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
    return jsonify(response)

if __name__ == '__main__':
    load_data()
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)