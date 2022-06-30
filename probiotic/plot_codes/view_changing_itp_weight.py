import pickle
from matplotlib import pyplot as plt, font_manager as fm
import numpy as np

with open('pickle_data/change_itp_weight.pkl', 'rb') as pf1:
    data = pickle.load(pf1)
    pass

font_text = {'family': 'Times New Roman', 'size': 20}
font_title = {'family': 'Times New Roman', 'size': 21}
font_tick = {'family': 'Times New Roman', 'size': 19}
font_legend = {'family': 'Times New Roman', 'size': 24}
w = np.linspace(0, 1, 21)
fig = plt.figure(1)
plt.plot(w, data['nm_trains'], 's-', label=r'$R^2_{nm,train}$')
plt.plot(w, data['nm_valis'], '>-', label='nm vali')
plt.plot(w, data['lg_trains'], 'v-', label='lg train')
plt.plot(w, data['lg_valis'], 'p-', label='lg vali')
legend = plt.legend(loc='lower right', prop=font_legend, frameon=False)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')  # 设置图例legend背景透明
plt.xticks(np.linspace(0, 1, 6),
           fontproperties=fm.FontProperties(**font_tick))
# plt.yticks(np.linspace(0.88, 0.98, 6),
#            fontproperties=fm.FontProperties(**font_tick))
plt.xlabel('Interpolation weight', fontdict=font_text)
plt.ylabel('Accuracy', fontdict=font_text)
plt.tight_layout()
plt.show()
fig.savefig('figures/changing_itp_weight.png', transparent=True)  # 设置整体透明
