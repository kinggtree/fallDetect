# history_data_pool_new.py

import numpy as np
import threading
import joblib
import glob
import os
import re # 导入 re 用于排序

# --- 配置 ---
# 【修改】指向包含所有受试者npy文件的文件夹
SUBJECT_DATA_DIR = ".\\Processed_Per_Subject"
# 【【【新增】】】 指向包含所有受试者特征文件的文件夹
FEATURES_DIR = ".\\Extracted_Features_Per_Subject" 
SCALER_PATH = "autoregression_timeseries_data_scaler.save"

# --- 全局变量 ---
ALL_SUBJECT_DATA = []   # 【修改】将是一个列表，每个元素是一个受试者的 (N, 200, 11) 数组
ALL_SUBJECT_FEATURES = [] # 【【【新增】】】
ALL_SUBJECT_LABELS = []   # 【【【新增】】】
DATA_ITEM_SHAPE = None  # 用于存储单个数据项的形状，例如 (200, 11)
CURRENT_SUBJECT_INDEX = 0  # 【修改】跟踪当前是第几个受试者
CURRENT_INDEX_IN_SUBJECT = 0 # 【修改】跟踪在当前受试者数据中的索引
SEND_REAL_DATA = True   # 控制标志，默认为发送真实数据
reset_timer = None # 用于跟踪重置定时器
data_lock = threading.Lock() # 确保线程安全

def load_and_prepare_data():
    global ALL_SUBJECT_DATA, ALL_SUBJECT_FEATURES, ALL_SUBJECT_LABELS, DATA_ITEM_SHAPE
    
    # --- 1. 加载缩放器 ---
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Loaded scaler from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Scaler file not found at {SCALER_PATH}. Exiting.")
        exit(1)

    # --- 2. 查找并排序受试者文件 ---
    search_path = os.path.join(SUBJECT_DATA_DIR, "SensorDataSequences_sub*.npy")
    subject_files = glob.glob(search_path)
    
    if not subject_files:
        print(f"ERROR: No 'SensorDataSequences_sub*.npy' files found in {SUBJECT_DATA_DIR}. Exiting.")
        exit(1)
        
    def get_subject_num(filepath):
        match = re.search(r'sub(\d+)\.npy', os.path.basename(filepath))
        return int(match.group(1)) if match else -1
        
    sorted_files = sorted(subject_files, key=get_subject_num)
    
    print(f"Found and sorted {len(sorted_files)} subject data files.")

    # --- 3. 逐个加载、缩放并存储所有数据 (原始, 特征, 标签) ---
    for raw_data_filepath in sorted_files:
        base_name = os.path.basename(raw_data_filepath)
        subject_id_str = re.search(r'sub(\d+)', base_name).group(1)
        
        # 构建对应的特征和标签文件路径
        feature_filepath = os.path.join(FEATURES_DIR, f"Features_sub{subject_id_str}.npy")
        label_filepath = os.path.join(SUBJECT_DATA_DIR, f"SensorLabelSequences_sub{subject_id_str}.npy")

        print(f"\nProcessing subject {subject_id_str}...")

        try:
            # 加载原始数据、特征和标签
            raw_data = np.load(raw_data_filepath)
            features = np.load(feature_filepath)
            labels = np.load(label_filepath)

            # --- 验证数据一致性 ---
            if not (raw_data.shape[0] == features.shape[0] == labels.shape[0]):
                print(f"  WARNING: Skipping subject {subject_id_str} due to data mismatch.")
                print(f"    Raw sequences: {raw_data.shape[0]}, Features: {features.shape[0]}, Labels: {labels.shape[0]}")
                continue

            if raw_data.shape[0] == 0:
                print(f"  WARNING: Skipping subject {subject_id_str}, contains 0 sequences.")
                continue

            # 归一化处理原始数据
            scaled_subject_data = []
            for i in range(raw_data.shape[0]):
                scaled_sequence = scaler.transform(raw_data[i])
                scaled_subject_data.append(scaled_sequence)
            
            # 将这个受试者的所有数据添加到全局列表
            ALL_SUBJECT_DATA.append(np.array(scaled_subject_data))
            ALL_SUBJECT_FEATURES.append(features)
            ALL_SUBJECT_LABELS.append(labels)
            
            print(f"  Loaded and scaled {raw_data.shape[0]} sequences, features, and labels.")

        except FileNotFoundError as e:
            print(f"  WARNING: Skipping subject {subject_id_str}. File not found: {e.filename}")
            continue

    if not ALL_SUBJECT_DATA:
        print("ERROR: No data was successfully loaded. Exiting.")
        exit(1)

    # 存储单个数据项的形状
    DATA_ITEM_SHAPE = ALL_SUBJECT_DATA[0].shape[1:]
    print(f"\nData item shape set to: {DATA_ITEM_SHAPE}")
    print(f"History data pool is ready. {len(ALL_SUBJECT_DATA)} subjects loaded.")


# --- 重置索引的回调函数 ---
def reset_current_index():
    """因超时而重置索引（返回到第一个受试者的第一条数据）"""
    global CURRENT_SUBJECT_INDEX, CURRENT_INDEX_IN_SUBJECT
    with data_lock:
        # 检查索引是否已经为0，避免重复打印消息
        if CURRENT_SUBJECT_INDEX != 0 or CURRENT_INDEX_IN_SUBJECT != 0:
            CURRENT_SUBJECT_INDEX = 0
            CURRENT_INDEX_IN_SUBJECT = 0
            # 在打印前后添加换行符，使其在服务器日志中更显眼
            print("\n--- 5 seconds of inactivity. Resetting to Subject 0, Index 0. ---\n")


# ===================================================================
# --- 发送数据切片的函数 ---
# ===================================================================

def get_raw_data_slice_direct():
    """
    【修改】直接获取当前受试者的单条数据块，并附带 'is_done' 标志。
    返回: (response_dict, status_code)
    """
    global CURRENT_SUBJECT_INDEX, CURRENT_INDEX_IN_SUBJECT, SEND_REAL_DATA
    with data_lock:
        
        # 检查受试者索引是否越界
        if CURRENT_SUBJECT_INDEX >= len(ALL_SUBJECT_DATA):
            return ({"error": "End of all subject data"}, 404)
            
        # 获取当前受试者的所有数据
        current_subject_data = ALL_SUBJECT_DATA[CURRENT_SUBJECT_INDEX]
        
        # 检查当前受试者的数据是否已发完
        if CURRENT_INDEX_IN_SUBJECT >= len(current_subject_data):
            msg = f"End of data for subject index {CURRENT_SUBJECT_INDEX}. Call next_subject to advance."
            return ({"error": msg}, 404)

        # 默认 'is_done' 为 False
        is_done = False

        # 根据 SEND_REAL_DATA 标志决定发送什么
        if SEND_REAL_DATA:
            # 提取真实数据块
            data_slice = current_subject_data[CURRENT_INDEX_IN_SUBJECT]
        else:
            # 创建一个全零的数据块
            zero_shape = DATA_ITEM_SHAPE
            data_slice = np.zeros(zero_shape)

        # 无论发送什么，索引都照常更新
        CURRENT_INDEX_IN_SUBJECT += 1
        
        # 【关键】检查是否是该受试者的最后一条数据
        if CURRENT_INDEX_IN_SUBJECT == len(current_subject_data):
            is_done = True
            print(f"--- Just served the last sequence for subject index {CURRENT_SUBJECT_INDEX}. Setting is_done=True. ---")
        
        response = {
            "data_slice": data_slice.tolist(),
            "is_done": is_done
        }
    
    # 模拟 200 OK
    return (response, 200)

def set_instruction_direct(instruction):
    """
    直接设置指令，模拟 Flask 路由的逻辑。
    返回: (response_dict, status_code)
    """
    global SEND_REAL_DATA
    try:
        with data_lock:
            if instruction == 1 or instruction is True:
                SEND_REAL_DATA = True
                message = "Instruction set to: SEND REAL DATA"
            elif instruction == 0 or instruction is False:
                SEND_REAL_DATA = False
                message = "Instruction set to: SEND ZERO DATA"
            else:
                # 模拟 400 Bad Request
                return ({"error": "Invalid instruction. Send 1 (true) or 0 (false)."}, 400)
        
        # print(message) # 在服务器日志中打印状态
        # 模拟 200 OK
        return ({"status": "success", "message": message}, 200)

    except Exception as e:
        # 模拟 400 Bad Request (或 500 Internal Server Error)
        return ({"error": str(e)}, 400)

# --- 【【【新增函数】】】 ---
def next_subject_direct():
    """
    (新) 切换到下一个受试者的数据。
    当 model_runner 收到 'is_done: True' 时，应调用此函数。
    """
    global CURRENT_SUBJECT_INDEX, CURRENT_INDEX_IN_SUBJECT
    with data_lock:
        # 推进到下一个受试者
        CURRENT_SUBJECT_INDEX += 1
        # 将该受试者的索引重置为0
        CURRENT_INDEX_IN_SUBJECT = 0
        
        if CURRENT_SUBJECT_INDEX >= len(ALL_SUBJECT_DATA):
            # 所有受试者都已完成
            print("--- Attempted to move to next subject, but all subjects are finished. ---")
            return ({"status": "error", "message": "No more subjects"}, 404)
        else:
            print(f"\n--- Advanced to subject index: {CURRENT_SUBJECT_INDEX} ---\n")
            return ({"status": "success", "message": f"Moved to subject index {CURRENT_SUBJECT_INDEX}"}, 200)
        

# ===================================================================
# --- 发送特征的函数 ---
# ===================================================================
def get_feature_direct():
    """
    (新) 获取与上一个原始数据块相对应的特征和标签。
    此函数应在调用 get_raw_data_slice_direct() 之后调用。
    返回: (response_dict, status_code)
    """
    with data_lock:
        # 检查受试者索引是否有效
        if CURRENT_SUBJECT_INDEX >= len(ALL_SUBJECT_FEATURES):
            return ({"error": "Subject index out of bounds"}, 404)
        
        # get_raw_data_slice_direct() 会使索引加1，所以我们需要用 -1 来获取刚刚发送的数据对应的特征
        feature_index = CURRENT_INDEX_IN_SUBJECT - 1

        # 检查特征索引是否有效 (例如，在请求任何原始数据之前调用此函数)
        if feature_index < 0:
            return ({"error": "Invalid feature index. Call get_raw_data_slice_direct first."}, 400)

        # 获取当前受试者的特征和标签数据
        current_features = ALL_SUBJECT_FEATURES[CURRENT_SUBJECT_INDEX]
        current_labels = ALL_SUBJECT_LABELS[CURRENT_SUBJECT_INDEX]

        # 检查特征索引是否越界 (理论上不应发生，但作为安全检查)
        if feature_index >= len(current_features):
            return ({"error": "Feature index out of bounds for the current subject"}, 404)

        feature = current_features[feature_index]
        label = current_labels[feature_index]

        response = {
            "feature": feature.tolist(),
            "label": int(label) # 确保标签是标准的int类型
        }

    # 模拟 200 OK
    return (response, 200)