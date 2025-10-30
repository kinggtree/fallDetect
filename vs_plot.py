import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 执行筛选操作 ---

# 定义文件名
output_file1 = 'DQN_evaluate_lazySync_xishu.csv'
output_file2 = 'DQN_evaluate_ours_xishu.csv'

# 定义绘图函数中使用的颜色
colors = {
    'blue': '#1f77b4',  # Matplotlib 默认蓝色
    'red': '#d62728'    # Matplotlib 默认红色
}

try:
    df1_filtered = pd.read_csv(output_file1)
    df2_filtered = pd.read_csv(output_file2)

    # --- 2. 定义修改后的绘图函数 ---

    def plot_lazy_verser_act_from_data(df_lazy, df_ours):
        
        print("开始绘图...")
        
        # 检查筛选后的数据是否为空
        if df_lazy.empty or df_ours.empty:
            print("错误：筛选后的数据为空，无法进行绘图。")
            return

        try:
            # 窗口越大，曲线越平滑，但响应越慢
            window_size = 50
            
            # 对 lazy_values (Cumulative_Accuracy) 计算滑动平均
            df_lazy['Cumulative_Accuracy_MA'] = df_lazy['Cumulative_Accuracy'].rolling(window=window_size, min_periods=1).mean()
            
            # 对 act_values (Action_1_Ratio) 计算滑动平均
            df_ours['Action_1_Ratio_MA'] = df_ours['Action_1_Ratio'].rolling(window=window_size, min_periods=1).mean()
            
            print(f"已计算窗口大小为 {window_size} 的滑动平均值。")
            
            x_values = df_lazy['Current_Step']
            
            lazy_values_ma = df_lazy['Cumulative_Accuracy_MA']
            
            act_values_ma = df_ours['Action_1_Ratio_MA']
            
            print(f"已提取 'Current_Step', 'Cumulative_Accuracy_MA' 和 'Action_1_Ratio_MA' 用于绘图。")


        except KeyError as e:
            print(f"错误：在CSV文件中找不到预期的列。")
            print(f"详细错误: {e}")
            return

        # --- 开始绘图 (使用平滑后的数据) ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left axis for accuracy (LazySyncDT)
        line1 = ax1.plot(x_values, lazy_values_ma, marker='o', color=colors['blue'], linewidth=2, markersize=8, label='accuracy (LazySyncDT, MA)')
        ax1.set_ylabel('accuracy (LazySyncDT, MA)', fontsize=12, color=colors['blue']) # 更新 Y 轴标签
        ax1.tick_params(axis='y', labelcolor=colors['blue'])

        # Right axis for sync. frequency (ActSyncDT)
        ax2 = ax1.twinx()
        line2 = ax2.plot(x_values, act_values_ma, marker='s', color=colors['red'], linewidth=2, markersize=8, label='sync. frequency (ActSyncDT/Ours, MA)')
        ax2.set_ylabel('sync. frequency (ActSyncDT/Ours, MA)', fontsize=12, color=colors['red']) # 更新 Y 轴标签
        ax2.tick_params(axis='y', labelcolor=colors['red'])

        # Labels and title
        ax1.set_xlabel('time slots', fontsize=12)
        ax1.set_xticks([]) 

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=10)
        
        plt.title('LazySyncDT vs ActSyncDT (Ours) - Moving Average', fontsize=14) # 更新标题
        fig.tight_layout() 

        # Save plot
        output_image_file = "lazy_vs_act_moving_average.png"
        plt.savefig(output_image_file, dpi=400, bbox_inches='tight')
        print(f"绘图完成，已保存为 '{output_image_file}'")

    plot_lazy_verser_act_from_data(df1_filtered, df2_filtered)


except FileNotFoundError:
    print(f"错误：找不到 {output_file1} 或 {output_file2}。")
except Exception as e:
    print(f"执行过程中发生意外错误: {e}")