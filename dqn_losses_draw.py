import pandas as pd
import io
import matplotlib.pyplot as plt

FILE_PATH = "model_runner_dqn_log_20251026_155720.csv"

# 加载 CSV 数据
df = pd.read_csv(FILE_PATH)  # 请替换为实际文件名

# 创建一个 'Step' 列（使用索引）作为 X 轴
df['Step'] = df.index

# --- Matplotlib 绘图代码 ---

# 创建一个图形和两个子图 (ax1, ax2)，上下排列，共享 x 轴
# figsize=(10, 8) 设置图形大小
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 子图 1: 绘制 DQN Loss 曲线
ax1.plot(df['Step'], df['DQN Loss'], marker='o', linestyle='-', color='r')
ax1.set_title('DQN Training Loss Curve') # 设置标题
ax1.set_ylabel('DQN Loss') # 设置 y 轴标签
ax1.grid(True) # 显示网格

# 子图 2: 绘制 Cumulative Accuracy 曲线
ax2.plot(df['Step'], df['Cumulative_Accuracy'], marker='s', linestyle='-', color='b')
ax2.set_title('Cumulative Accuracy Curve') # 设置标题
ax2.set_ylabel('Cumulative Accuracy (%)') # 设置 y 轴标签
ax2.set_xlabel('Training Step') # 设置 x 轴标签
ax2.grid(True) # 显示网格

# 自动调整布局，防止标签重叠
plt.tight_layout()

# 保存图像到文件
plot_filename = f"training_curves_{FILE_PATH[21:]}.png"
plt.savefig(plot_filename)

print(f"图像已保存到 {plot_filename}")

# --- 绘图代码结束 ---