# 网络配置
HOST = '127.0.0.1'  # 本地主机
PORT = 65432        # 一个未被占用的端口

# 数据流配置

TARGET_SAMPLING_RATE_HZ = 50.0  # Target sampling rate in Hz
TARGET_SAMPLING_PERIOD = f"{int(1000 / TARGET_SAMPLING_RATE_HZ)}ms"
SEQUENCE_LENGTH = int(TARGET_SAMPLING_RATE_HZ * 4) # 200 samples for 4 seconds at 50Hz
STEP = int(TARGET_SAMPLING_RATE_HZ * 1)          # 50 samples for 1 second step at 50Hz
DATASET_PATH = 'MobiFall_Dataset'


SENSOR_CODES = ["acc", "gyro", "ori"]
EXPECTED_COLUMNS = {
    "acc": ["acc_x", "acc_y", "acc_z"],
    "gyro": ["gyro_x", "gyro_y", "gyro_z"],
    "ori": ["ori_azimuth", "ori_pitch", "ori_roll"]
}
ALL_FEATURE_COLUMNS = [
    "acc_x", "acc_y", "acc_z", "acc_smv",
    "gyro_x", "gyro_y", "gyro_z", "gyro_smv",
    "ori_azimuth", "ori_pitch", "ori_roll"
]