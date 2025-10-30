import pandas as pd
import numpy as np


def analyze_data(data_frame: pd.DataFrame):
    try:
        df = data_frame

        # 1. 计算 'Zero_Vectors_Ratio' 和 'Cumulative_Accuracy' 的平均值
        mean_zero_vectors = df['Zero_Vectors_Ratio'].mean()
        mean_cumulative_accuracy = df['Cumulative_Accuracy'].mean()

        print(f"--- 基础统计分析 ---")
        print(f"'Zero_Vectors_Ratio' 的平均值: {mean_zero_vectors:.4f}")
        print(f"'Cumulative_Accuracy' 的平均值 (包含所有值): {mean_cumulative_accuracy:.4f}")

        # 2. 深入分析 'Cumulative_Accuracy' (解决用户担忧)
        total_rows = len(df)
        accuracy_100_count = (df['Cumulative_Accuracy'] == 100.0).sum()
        accuracy_100_percentage = (accuracy_100_count / total_rows) * 100

        print(f"\n--- 'Cumulative_Accuracy' 深度分析 ---")
        print(f"总数据行数: {total_rows}")
        print(f"值为 100.0 的行数: {accuracy_100_count}")
        print(f"值为 100.0 的行所占百分比: {accuracy_100_percentage:.2f}%")

        # 3. 计算排除了100.0之后的 'Cumulative_Accuracy' 平均值
        accuracy_not_100 = df[df['Cumulative_Accuracy'] != 100.0]['Cumulative_Accuracy']
        
        if len(accuracy_not_100) > 0:
            mean_accuracy_not_100 = accuracy_not_100.mean()
            print(f"排除了100.0后, 'Cumulative_Accuracy' 的平均值: {mean_accuracy_not_100:.4f}")
        else:
            print("所有 'Cumulative_Accuracy' 的值均为 100.0。")

        # 4. (可选) 提供中位数，它对极值不敏感
        median_zero_vectors = df['Zero_Vectors_Ratio'].median()
        median_cumulative_accuracy = df['Cumulative_Accuracy'].median()
        
        print(f"\n--- 中位数 (对偏态数据更具参考性) ---")
        print(f"'Zero_Vectors_Ratio' 的中位数: {median_zero_vectors:.4f}")
        print(f"'Cumulative_Accuracy' 的中位数: {median_cumulative_accuracy:.4f}")

    except KeyError as e:
        print(f"错误：找不到列 {e}。请检查CSV文件中的列名。")
    except Exception as e:
        print(f"分析数据时出错: {e}")
