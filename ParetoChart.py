import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. 设置matplotlib以正确显示英文（默认通常就是英文，所以不需要特别的字体设置）
plt.rcParams['font.sans-serif'] = ['Arial']  # 尝试使用更常见的英文字体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

# 2. 您的原始数据
data_text = """
方法,采样率,准确率
random,0.25,92.1444
random,0.5,95.9033
random,0.75,97.9200
uniform,0.25,91.3314
uniform,0.5,94.1400
uniform,0.75,96.4734
lazySync,0,87.8049
cloud,1,98.7330
ours,0.118,98.2578
"""

# 3. 将数据读入Pandas DataFrame
df = pd.read_csv(io.StringIO(data_text))

# 4. **添加去除虚线的开关**
show_grid = True # 设置为 False 将不显示网格线

# 5. 准备绘图
fig, ax = plt.subplots(figsize=(12, 8))

# 6. 为五种方法定义不同的颜色
colors = {
    'random': 'blue',
    'uniform': 'green',
    'lazySync': 'red',
    'cloud': 'purple',
    'ours': 'orange'
}

# 7. 绘制 "random" 和 "uniform" 的线
for method in ['random', 'uniform']:
    subset = df[df['方法'] == method].sort_values('采样率')
    ax.plot(subset['采样率'], subset['准确率'], 
            marker='o',             
            linestyle='-',          
            color=colors[method],   
            label=method)           
            
    last_point = subset.iloc[-1]
    ax.text(last_point['采样率'] + 0.01,  
            last_point['准确率'],       
            method,                       
            color=colors[method],         
            fontsize=12,                  
            ha='left',                    
            va='center')                  

# 8. 绘制 "lazySync", "cloud", "ours" 的点
single_points = df[df['方法'].isin(['lazySync', 'cloud', 'ours'])]

for idx, row in single_points.iterrows():
    method = row['方法']
    
    ax.scatter(row['采样率'], row['准确率'], 
               color=colors[method],   
               label=method,           
               s=100,                  
               zorder=5)               
               
    if method == 'cloud':
        x_offset = -0.01
        ha = 'right'  
    else:
        x_offset = 0.01
        ha = 'left'   
        
    ax.text(row['采样率'] + x_offset,  
            row['准确率'],              
            method,                     
            color=colors[method],       
            fontsize=12,                
            ha=ha,                      
            va='center')                

# 9. 美化图表 - **所有文本改为英文**
ax.set_xlabel('Sampling Rate', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Accuracy vs. Sampling Rate Comparison of Different Methods', fontsize=16)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(85, 100)

# **根据开关决定是否显示网格线**
if show_grid:
    ax.grid(True, linestyle='--', alpha=0.7)

ax.legend(title='Method', loc='lower right') # 图例标题也改为英文

plt.tight_layout()

# 10. 保存图像
plt.savefig('pareto_plot.png') # 保存为不同的文件名，以区分

print("图像已成功保存为 'pareto_plot.png'")