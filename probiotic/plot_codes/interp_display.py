import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import font_manager as fm

font_text = {'family': 'Times New Roman', 'size': 20}
font_title = {'family': 'Times New Roman', 'size': 21}
font_tick = {'family': 'Times New Roman', 'size': 19}
font_legend = {'family': 'Times New Roman', 'size': 24}
df = pd.read_excel('excel/itp_ft_s.xlsx', sheet_name='Sheet1')
t = df['t(s)'].values[25:95]
s = df['itp'].values[25:95]
real_ind = df['tag'].values[25:95]
fig = plt.figure(1)
# fig.patch.set_alpha(0.)
plt.scatter(t[real_ind], s[real_ind], s=60, marker='o', label='observed')
plt.scatter(t[~real_ind], s[~real_ind], s=6, marker='s', label='interpolated')
plt.xlim([25, 95])
plt.xticks(np.arange(25, 96, 10),
           fontproperties=fm.FontProperties(**font_tick))
plt.yticks(np.arange(0.88, 1.005, 0.02),
           fontproperties=fm.FontProperties(**font_tick))
plt.ylim([0.875, 1.005])
plt.xlabel('Time (s)', fontdict=font_text)
plt.ylabel('Survival rate', fontdict=font_text)
legend = plt.legend(loc='best', prop=font_legend, frameon=False)  # 设置图例无边框
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')  # 设置图例legend背景透明
plt.tight_layout()
fig.savefig('testhh.png', transparent=True)  # 设置整体透明
plt.show()
