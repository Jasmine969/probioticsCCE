import pickle
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

ta_base = 90
va_base = 0.7
ws_base = 0.1
vd_base = 1.5e-9
bases = {'Ta': 90, 'va': 0.7, 'ws': 0.1, 'vd': 1.5e-9}
with open('../../pickle_data/sns-ana-time-0d1.pkl', 'rb') as pickle_file:
    dct_rate, dct_cond = pickle.load(pickle_file)
# list to ndarray
dct_rate = {key: np.asarray(val) for key, val in dct_rate.items()}
dct_cond = {key: np.asarray(val) for key, val in dct_cond.items()}
relative_rate = {}
relative_cond = {}
for cond_name, cond_val in dct_cond.items():
    # calc relative change of cond and val
    relative_cond[cond_name] = (cond_val - bases[cond_name]) / cond_val[0]
    relative_rate[cond_name] = (dct_rate[cond_name] - dct_rate[cond_name][0]
                                ) / dct_rate[cond_name][0]
# 绘图
plt.rc('font', family='Times New Roman', size=17)
font_formula = fm.FontProperties(math_fontfamily='cm', size=21)
font_text = fm.FontProperties(family='Times New Roman', size=21)
font_legend = {'size': 19, 'math_fontfamily': 'cm'}
markers = ['.-', 's-', 'd-', 'v-']
labels = [r'$T_\mathrm{a}$', r'$v_\mathrm{a}$', r'$w_\mathrm{s}$', r'$V_\mathrm{d}$']
for ind, cond_name in enumerate(dct_rate.keys()):
    plt.plot(relative_cond[cond_name], relative_rate[cond_name],
             markers[ind], label=labels[ind])
plt.xlabel('Relative change of drying conditions', fontproperties=font_text)
plt.ylabel(r'Relative change of survival rate', fontproperties=font_text)
plt.legend(prop=font_legend)
plt.gcf().set_size_inches([7.84, 5.7])
plt.gcf().savefig('../figures/slopes-sns-ana.png', transparent=True)
