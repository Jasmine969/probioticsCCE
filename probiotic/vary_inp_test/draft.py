import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = 'SimHei'
dataset = pd.DataFrame(data=[[5, 6, 8, 6],
                             [9, 6, 5, 7],
                             [4, 8, 9, 8],
                             [7, 8, 20, 7],
                             [8, 7, 6, 7]],
                       index=['表达与沟通', '团队合作', '分析与综合信息 ', '创新思维', '批判性思维'],
                       columns=['金融学院', '文化传媒学院', '信息学院 ', '管理学院 '])
radar_labels = dataset.index
nAttr = 5
data = dataset.values  # 数据值
data_labels = dataset.columns
# 设置角度
angles = np.linspace(0, 2 * np.pi, nAttr,
                     endpoint=False)
# data = np.concatenate((data, [data[0]]))
data = np.vstack((data, [data[0]]))
angles = np.concatenate((angles, [angles[0]]))
# 设置画布
fig = plt.figure(facecolor="white", figsize=(10, 6))
plt.subplot(111, polar=True)  # 设置极坐标
# 绘图
plt.plot(angles, data, 'o-',
         linewidth=1.5, alpha=0.2)
# 填充颜色
plt.fill(angles, data, alpha=0.25)
plt.thetagrids(angles[:-1] * 180 / np.pi,
               radar_labels)
plt.figtext(0.52, 0.95, '大学生通识能力分析',
            ha='center', size=20)
# 设置图例
legend = plt.legend(data_labels,
                    loc=(1.1, 0.05),
                    labelspacing=0.1)
plt.setp(legend.get_texts(),
         fontsize='large')
plt.grid(True)
# plt.savefig('tongshi.png')
plt.show()
