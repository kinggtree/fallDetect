import matplotlib.pyplot as plt
import pandas as pd

colors = {
    "yellow": "#D6B656",
    "orange": "#D79B00",
    "red": "#B85450",
    "green": "#82B366",
    "blue": "#6C8EBF",
    "purple": "#9673A6",
}


def plot_u_verses_f_mobifall():
    plt.figure(figsize=(6, 6))

    f1 = [1]
    u1 = [98.7330]
    label1 = 'Cloud'

    # 更新 'Random' 的数据
    f2 = [0.2, 0.4, 0.6, 0.8]
    u2 = [91.6165, 94.5201, 97.0753, 98.1100]
    label2 = 'Random'

    # 更新 'Uniform' 的数据
    f3 = [0.2, 0.4, 0.6, 0.8]
    u3 = [90.6979, 93.1158, 94.9319, 96.9486]
    label3 = 'Uniform'

    f4 = [0]
    u4 = [87.8049]
    label4 = 'LazySyncDT'

    f5 = [0.118]
    u5 = [98.2578]
    label5 = 'ActSyncDT'

    # Create scatter plot
    plt.scatter(f1, u1, color=colors['purple'], marker='s', label=label1)
    plt.plot(f2, u2, color=colors['yellow'], marker='^', label=label2)
    plt.plot(f3, u3, color=colors['green'], marker='v', label=label3)
    plt.scatter(f4, u4, color=colors['blue'], marker='x', label=label4)
    plt.scatter(f5, u5, color=colors['red'], marker='o', label=label5)
    plt.legend(loc='best', fontsize=16)

    # Labels and title
    plt.xlabel('f (sync. frequency)', fontsize=16)
    plt.ylabel('u (accuracy)', fontsize=16)

    # Show plot
    plt.savefig("u_verses_f_mobifall.png", dpi=400, bbox_inches='tight')


def plot_u_verses_f_umafall():
    plt.figure(figsize=(6, 6))

    f1 = [1]
    u1 = [98.8185]
    label1 = 'Cloud'

    # 'Random' 的数据已更新
    f2 = [0.2, 0.4, 0.6, 0.8]
    u2 = [96.4196, 97.5653, 98.1919, 98.6216]
    label2 = 'Random'

    # 'Uniform' 的数据已更新
    f3 = [0.2, 0.4, 0.6, 0.8]
    u3 = [95.9542, 96.4017, 97.0820, 97.8518]
    label3 = 'Uniform'

    f4 = [0]
    u4 = [95.3455]
    label4 = 'LazySyncDT'

    f5 = [0.0462]
    u5 = [97.2431]
    label5 = 'ActSyncDT'

    # Create scatter plot
    plt.scatter(f1, u1, color=colors['purple'], marker='s', label=label1)
    plt.plot(f2, u2, color=colors['yellow'], marker='^', label=label2)
    plt.plot(f3, u3, color=colors['green'], marker='v', label=label3)
    plt.scatter(f4, u4, color=colors['blue'], marker='x', label=label4)
    plt.scatter(f5, u5, color=colors['red'], marker='o', label=label5)
    plt.legend(loc='best', fontsize=16)

    # Labels and title
    plt.xlabel('f (sync. frequency)', fontsize=16)
    plt.ylabel('u (accuracy)', fontsize=16)

    # Show plot
    plt.savefig("u_verses_f_umafall.png", dpi=400, bbox_inches='tight')


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


def main():
    plot_u_verses_f_mobifall()
    plot_u_verses_f_umafall()

    df_lazy = pd.read_csv("DQN_evaluate_lazySync.csv")
    df_ours = pd.read_csv("DQN_evaluate_ours.csv")
    plot_lazy_verser_act_from_data(df_lazy, df_ours)


if __name__ == '__main__':
    main()
