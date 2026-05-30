import matplotlib.pyplot as plt
import os

# 数据
methods = ['After stage 2', 'After input', 'After stage 3', 'w/o L_causal', 'w/o L_sparse']
rie_values = [75.07, 75.99, 68.37, 64.40, 66.36]  # 百分比

# 配色，仿照上传图片中的 RIE 柱子颜色
colors = ['#5F9EA0'] * len(rie_values)  # 青绿色系

# 创建 output 文件夹（如果不存在）
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# 创建柱状图
plt.figure(figsize=(8, 5))
# bars = plt.bar(methods, rie_values, color=colors, width=0.3)

# ====== 这里替换原来的 plt.bar() ======
import numpy as np
x = np.arange(len(methods))  # [0,1,2,3,4]
spacing = 0.2  # 调整整体间距缩放
bars = plt.bar(x * spacing, rie_values, color=colors, width=0.1)  

plt.xticks(x * spacing, methods)  # 设置标签
# ======================================

# 添加数值标签
for bar, value in zip(bars, rie_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{value:.2f}%', ha='center', va='bottom', fontsize=10)

# 设置纵轴标签
plt.ylabel('RIE (%)', fontsize=12)
plt.ylim(60, 90)  # 纵轴从50%开始
plt.yticks(range(60, 91, 10))  # 每10%一个分度

# 去掉顶部和右侧边框，更像论文风格
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.title('Information Entropy Ratio (RIE) for Different Methods', fontsize=12)
plt.tight_layout()

# 保存图像到 ./output 文件夹
output_path = os.path.join(output_dir, 'rie_bar_chart.png')
plt.savefig(output_path, dpi=300)
print(f'图像已保存到 {output_path}')

# 显示图像
plt.show()