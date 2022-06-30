from matplotlib import pyplot as plt, font_manager as fm
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = pd.read_excel('model_comparison.xlsx', usecols='A,D:G').dropna().to_numpy()
x = np.arange(14)
plt.rc('font', family='Times New Roman', size=15)
font_formula = fm.FontProperties(
    math_fontfamily='cm', size=20
)
font_text = {'size': 20}
# fig1, ax1 = plt.subplots()
# # fig2, ax2 = plt.subplots()
# # fig3, ax3 = plt.subplots()
# # fig4, ax4 = plt.subplots()
# # fig.set_size_inches([18.85, 7])

colors = ['#9999ff', '#ff9999', '#b6c9bb', '#f6bc65']
bar_width = 0.4
ind = 1

fig, axMain = plt.subplots()
fig.set_dpi(100)
fig.set_size_inches([15.36, 7.57])
axMain.bar(x, data[:, 2], color=colors[1], width=bar_width)
plt.yscale('linear')
axMain.set_ylim([1e-10, 1.1])
axMain.spines['bottom'].set_visible(False)
axMain.xaxis.set_ticks_position('top')
axMain.xaxis.set_visible(False)

divier = make_axes_locatable(axMain)
axLog = divier.append_axes("bottom", size=1.0, pad=0, sharex=axMain)
axLog.set_yscale('symlog')
axLog.set_ylim([-60, 1e-10])
axLog.set_yticks([-10, -1, 0])
axLog.set_yticklabels(['-10', '-1.0', '0.0'])
axLog.set_xticks(x)
axLog.set_xticklabels(x)
axLog.bar(x, data[:, 2], color=colors[1], width=bar_width)
axLog.spines['top'].set_visible(False)
axLog.xaxis.set_ticks_position('bottom')
plt.setp(axLog.get_xticklabels(), visible=True)
# 坐标轴名字在PPT手动添加
# axLog.set_xlabel('Model', fontdict=font_text, labelpad=28)
# axLog.set_ylabel(r'$R^2$', fontproperties=font_formula)
text_skip = 0.03
for i in range(14):
    curR2 = data[i, 1 + 1]
    if curR2 > 0:
        axMain.text(x[i], curR2 + text_skip, f'{curR2:.4}', ha='center')
    elif curR2 == 0:
        axLog.text(x[i], curR2 + text_skip,
                   'Divergence' if i == 8 else 'Unfitted', ha='center')
    else:
        axLog.text(x[i], curR2 - 2 ** (-curR2 / 4) if curR2 < -0.1 else curR2 - 0.35,
                   f'{curR2:.4}', ha='center', va='top')
fig.savefig('../figures/model-comp/bar2-log10.jpg')
