from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import font_manager as fm
from matplotlib.ticker import AutoMinorLocator
import pickle
from my_functions import is_pareto_efficient_simple, split_discontinuity
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import torch
from plotly import express as px, offline
from itertools import chain


def x2ws(x):
    return 1 / (x + 1) * 100


with open('../pickle_data/monte-carlo-dry-5w.pkl', 'rb') as pf:
    data_mc = pickle.load(pf)

# mode = 'zh'  # chinese
mode = 'eng'

# process experimental points
exp_cond = {
    'Ta': [70, 90, 110, 70, 90, 110, 70, 70, 70],
    'va': [0.75] * 6 + [0.45, 0.45, 1.00],
    'ws0': [0.1] * 3 + [0.2] * 6,
    'vd': [2] * 6 + [1, 2, 2],
}
exp_dur = [271, 271, 121, 301, 301, 121, 401, 401, 361]
with open('../pickle_data/test35_s_non-tilde.pkl', 'rb') as pf1:
    dict_exp = pickle.load(pf1)
    ft_s_tv = dict_exp['ft_s_tv']
    ft_scaler = dict_exp['ft_scalar']
with open('../pickle_data/test35_lgg_s.pkl', 'rb') as pf:
    ft_s_test = pickle.load(pf)
ft_s = torch.vstack(ft_s_tv + ft_s_test).numpy()[:, :-1]
ft_s[:, :3] = ft_scaler.inverse_transform(ft_s[:, :3])
ft_s[:, 2] = np.clip(1 / (ft_s[:, 2] + 1) * 100, a_min=0, a_max=100)
# add drying conditions
ft_s = np.hstack((ft_s, np.empty((ft_s.shape[0], 4))))
for ind, cond_val in enumerate(exp_cond.values()):
    ft_s[:, ind + 4] = np.array(list(chain(*(
        [cond_val[i]] * dur for i, dur in enumerate(exp_dur)))))
ft_s = ft_s[:, [2, 3, 4, 5, 6, 7, 0]]

data_mc[:, 4] *= 100
data_mc = data_mc[data_mc[:, 2] <= 90]
plt.rc('font', family='Times New Roman', size=24)
font_formula = dict(math_fontfamily='cm', size=30)
font_legend = dict(family='Times New Roman', size=24)
font_text = {'size': 34}
font_text_zh = {'size': 31, 'family': 'SimHei'}

pure_mc_xs = data_mc[:, [0, 1]]
pure_mc_xs[:, 0] = x2ws(pure_mc_xs[:, 0])
is_pareto_mc = is_pareto_efficient_simple(
    pure_mc_xs, constraint=([90, 100], [0.65, 1]))
data_mc[:, 0] = x2ws(data_mc[:, 0])
mc_pareto = data_mc[is_pareto_mc, :]
# non_pareto = data_mc[~is_pareto_mc, :]
# fig_pareto = plt.figure(figsize=(9, 7.4))
# plt.scatter(
#     data_mc[:, 0], data_mc[:, 1], facecolor='w',
#     edgecolor='#C7EAE4', alpha=0.8, s=80, marker='s',
# )
# plt.scatter(
#     mc_pareto[:, 0], mc_pareto[:, 1], s=60,
#     c='#FF0000', marker='*'
# )
# plt.scatter(
#     ft_s[:, 2], ft_s[:, -1],
#     c='#75D19F', alpha=0.8, s=40, marker='>',
# )
# # plt.legend(loc='best', prop=font_legend, handletextpad=0.06)
# plt.plot([90, 90], [0, 1], '--', color='k')
# plt.plot([0, 100], [0.65, 0.65], '--', color='k')
# plt.xlim([0, 100])
# plt.ylim([0, 1])

# fig = px.scatter(, x="sepal_width", y="sepal_length", color="species",
#                  size='petal_length', hover_data=['petal_width'])
col_names = ['w_s', 's', 'T_a', 'v_a', 'w_{s,0}', 'V_{d,0}', 't']
df_mc = pd.DataFrame(data_mc, columns=col_names)
df_mc['type'] = 'Model predictions'
df_exp = pd.DataFrame(ft_s, columns=col_names)
df_exp['type'] = 'Experimental measurements'
df_pareto = pd.DataFrame(mc_pareto, columns=col_names)
df_pareto['type'] = 'Model predictions (optimal results)'
df_total: pd.DataFrame = pd.concat([df_mc, df_exp, df_pareto])
# df_total = df_total.round({'w_s': 1, 's': 2, 'T_a': 0, 'v_a': 2, 'w_{s,0}': 1, 'V_{d,0}': 1, 't': 0})
fig = px.scatter(df_total, x='w_s', y='s',
                 color='type', color_discrete_sequence=['#B5E3DB', '#A4D0F4', '#FF0000'],
                 symbol='type', symbol_sequence=['square-open', 'triangle-right', 'star'],
                 hover_data=['T_a', 'v_a', 'w_{s,0}', 'V_{d,0}', 't'])
fig.update_layout(
    font=dict(
        family='Times New Roman',
        size=20
    ),
    xaxis_title='w_s (wt%)'
)
offline.plot(fig, filename='drying optimization result.html')
