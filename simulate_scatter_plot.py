import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 请将 'your_file.csv' 替换成您上传的CSV文件名
try:
    df = pd.read_csv('simulate_log.csv')

    # 创建散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Zero_Vectors_Ratio', y='Probability', hue='Result', palette='coolwarm', alpha=0.7, s=50)
    
    # 添加标题和标签
    plt.title('Scatter Plot of Probability vs. Zero_Vectors_Ratio by Result', fontsize=16)
    plt.xlabel('Zero Vectors Ratio', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(True)
    plt.legend(title='Result')

    # 保存图表
    plt.savefig('scatter_plot.png')

    print("散点图已生成并保存为 scatter_plot.png")

except FileNotFoundError:
    print("找不到csv文件")
except KeyError as e:
    print(f"错误：CSV文件中缺少必要的列：{e}。请检查列名是否为 'Zero_Vectors_Ratio', 'Probability', 'Result'")