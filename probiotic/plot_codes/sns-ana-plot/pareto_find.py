from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import font_manager as fm
from matplotlib.ticker import AutoMinorLocator
import pickle
from probioticsCCE.my_functions import is_pareto_efficient_simple

data_optuna = pd.read_excel('../../vary_inp_test/optuna_res_dry/opt2.xlsx',
                            usecols='B,C,F:J').dropna().to_numpy()
with open('../../pickle_data/monte-carlo-dry-ta90-dur200.pkl', 'rb') as pf1:
    data_mc = pickle.load(pf1)

mode = 'zh'  # chinese
data_mc = data_mc[data_mc[:, 0] <= 1 / 0.7 - 1]  # ws>=70%
data_mc = data_mc[data_mc[:, 1] >= 0.01]  # s>=0.01
data_optuna = data_optuna[data_optuna[:, 0] <= 1 / 0.7 - 1]  # ws>=70%
data_optuna = data_optuna[data_optuna[:, 1] >= 0.01]  # s>=0.01

plt.rc('font', family='Times New Roman', size=20)
font_formula = dict(math_fontfamily='cm', size=28)
font_legend = dict(family='SimHei', size=20)
font_text = {'size': 27}
font_text_zh = {'size': 27, 'family': 'SimHei'}

pure_mc_xs = data_mc[:, [0, 1]]
pure_mc_xs[:, 1] *= -1
is_pareto_mc = is_pareto_efficient_simple(pure_mc_xs)
mc_pareto = data_mc[is_pareto_mc, :]

pure_optuna_xs = data_optuna[:, [0, 1]]
pure_optuna_xs[:, 1] *= -1
is_pareto_optuna = is_pareto_efficient_simple(pure_optuna_xs)
optuna_pareto = data_optuna[is_pareto_optuna, :]

fig_pareto = plt.figure(figsize=(9, 7.4))
plt.scatter(mc_pareto[:, 0], mc_pareto[:, 1], s=50,
            c='#1f77b4', marker='o')
plt.scatter(data_mc[~is_pareto_mc, 0], data_mc[~is_pareto_mc, 1],
            c='#ff7f0e', alpha=0.3, s=40, marker='o')
# plt.scatter(optuna_pareto[:, 0], optuna_pareto[:, 1], s=70,
#             c='#1f77b4', marker='s')
# plt.scatter(data_optuna[~is_pareto_optuna, 0], data_optuna[~is_pareto_optuna, 1],
#             c='#ff7f0e', alpha=0.3, s=60, marker='s')
plt.xlabel(r'$X~(\mathrm{kg/kg})$', fontdict=font_formula)
plt.ylabel(r'$s$', fontdict=font_formula)
plt.gcf().savefig('../../figures/vary_inp/pareto-optuna.png', transparent=True)
plt.close(fig_pareto)

optuna_pareto[:, 4] *= 100
mc_pareto[:, 4] *= 100
# hist of ta
var2col = {'ta': 2, 'va': 3, 'ws': 4, 'vd': 5, 't': 6}


def hist_scatter_var(
        data_pareto, var_name, xlabel,
        xlim, xticks=None, xticklabels=None
):
    fig_, ax_ = plt.subplots(figsize=(6.4, 1.12))
    plt.subplots_adjust(bottom=0.6)
    ax_.scatter(data_pareto[:, var2col[var_name]], np.random.random(data_pareto.shape[0]) / 4)
    ax_.set_xlim([xlim[0] - xlim[1] * 0.02, xlim[1] * 1.02])
    if xticks:
        ax_.set_xticks(xticks)
        ax_.set_xticklabels(xticklabels)
    ax_.spines['right'].set_visible(False)
    ax_.spines['left'].set_visible(False)
    ax_.spines['top'].set_visible(False)
    plt.tick_params(left=False)
    plt.tick_params(labelleft=False)
    ax_.set_xlabel(xlabel, fontdict=font_formula)
    return fig_, ax_


def hist_var(data_pareto, var_name, xlabel, bins,
             x_minor_num, xticks, yticks=None, y_minor_num=2):
    fig_, ax_ = plt.subplots(figsize=(7.1, 3.14))
    plt.subplots_adjust(bottom=0.279)
    ax_.grid(axis='y', which='major', alpha=0.5)
    x_minor_locator = AutoMinorLocator(x_minor_num)
    ax_.xaxis.set_minor_locator(x_minor_locator)
    y_minor_locator = AutoMinorLocator(y_minor_num)
    ax_.yaxis.set_minor_locator(y_minor_locator)
    ax_.hist(
        data_pareto[:, var2col[var_name]], bins=bins,
        # fill=False
        edgecolor='k', alpha=0.75, facecolor='g'
    )
    ax_.set_xlabel(xlabel, fontdict=font_formula)
    if mode == 'zh':
        ax_.set_ylabel('频数', fontdict=font_text_zh)
    else:
        ax_.set_ylabel('Count', fontdict=font_text)
    ax_.set_xticks(xticks)
    if yticks is not None:
        ax_.set_yticks(yticks)
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    return fig_, ax_


# mc hist
fig_ta_h, _ = hist_var(mc_pareto, 'ta', r'$T_\mathrm{a}~\mathrm{(^\circ\hspace{-0.25}C)}$',
                       x_minor_num=5, bins=list(range(60, 91, 2)),
                       xticks=list(range(60, 91, 5)),
                       yticks=[0,5,10,15])
fig_ta_h.savefig('../../figures/vary_inp/vary_ta_pareto-h.png', transparent=True)
fig_va_h, _ = hist_var(mc_pareto, 'va', r'$v_\mathrm{a}~\mathrm{(m/s)}$',
                       x_minor_num=2, bins=np.arange(0.2, 1.3, 0.1),
                       xticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
                       yticks=list(range(0, 6)))
fig_va_h.savefig('../../figures/vary_inp/vary_va_pareto-h.png', transparent=True)
fig_ws_h, _ = hist_var(mc_pareto, 'ws', r'$w_\mathrm{s}~\mathrm{(wt\%)}$',
                       x_minor_num=5, bins=list(range(10, 30, 2)),
                       xticks=[10, 15, 20, 25, 30],
                       yticks=list(range(0, 6)))
fig_ws_h.savefig('../../figures/vary_inp/vary_ws_pareto-h.png', transparent=True)
fig_vd_h, _ = hist_var(mc_pareto, 'vd', r'$V_\mathrm{d}~\mathrm{(\mu L)}$',
                       x_minor_num=5, bins=np.arange(0.5, 2.6, 0.2),
                       xticks=np.arange(0.5, 2.6, 0.5),
                       yticks=list(range(0, 16, 5)))
fig_vd_h.savefig('../../figures/vary_inp/vary_vd_pareto-h.png', transparent=True)
fig_t_h, _ = hist_var(mc_pareto, 't', r'$t~\mathrm{(s)}$',
                      x_minor_num=3, bins=list(range(50, 201, 10)),
                      xticks=list(range(50, 201, 30)),
                      yticks=list([0, 2, 4, 6]))
fig_t_h.savefig('../../figures/vary_inp/vary_t_pareto-h.png', transparent=True)

# # optuna hist
# fig_ta_op, _ = hist_var(optuna_pareto, 'ta', r'$T_\mathrm{a}~\mathrm{(^\circ\hspace{-0.25}C)}$',
#                         x_minor_num=5, bins=list(range(60, 76)),
#                         xticks=list(range(60, 76, 5)),
#                         yticks=[0, 10, 20, 30, 40], y_minor_num=2)
# fig_ta_op.savefig('../../figures/vary_inp/vary_ta_pareto-op.png', transparent=True)
# fig_va_op, _ = hist_var(optuna_pareto, 'va', r'$v_\mathrm{a}~\mathrm{(m/s)}$',
#                         x_minor_num=2, bins=np.arange(0.6, 1.3, 0.1),
#                         xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
#                         yticks=[0, 10, 20, 30], y_minor_num=2)
# fig_va_op.savefig('../../figures/vary_inp/vary_va_pareto-op.png', transparent=True)
# fig_ws_op, _ = hist_var(optuna_pareto, 'ws', r'$w_\mathrm{s}~\mathrm{(wt\%)}$',
#                         x_minor_num=4, bins=np.arange(10, 20.1, 0.5),
#                         xticks=[10, 12,14,16,18, 20],
#                         yticks=[0, 10, 20, 30, 40])
# fig_ws_op.savefig('../../figures/vary_inp/vary_ws_pareto-op.png', transparent=True)
# fig_vd_op, _ = hist_var(optuna_pareto, 'vd', r'$V_\mathrm{d}~\mathrm{(\mu L)}$',
#                         x_minor_num=2, bins=np.arange(0.5, 1.0, 0.05),
#                         xticks=np.arange(0.5, 1.1, 0.1),
#                         yticks=[0, 10, 20, 30, 40])
# fig_vd_op.savefig('../../figures/vary_inp/vary_vd_pareto-op.png', transparent=True)
# fig_t_op, _ = hist_var(optuna_pareto, 't', r'$t~\mathrm{(s)}$',
#                        x_minor_num=4, bins=list(range(125, 201, 5)),
#                        xticks=list(range(120, 201, 20)),
#                        yticks=[0, 5, 10])
# fig_t_op.savefig('../../figures/vary_inp/vary_t_pareto-op.png', transparent=True)

# mc scatter
# fig_ta_s, _ = hist_scatter_var('ta', r'$T_\mathrm{a}~\mathrm{(^\circ\hspace{-0.25}C)}$', [60, 90])
# fig_ta_s.savefig('../../figures/vary_inp/vary_ta_pareto-s.png', transparent=True)
# fig_va_s, _ = hist_scatter_var('va', r'$v_\mathrm{a}~\mathrm{(m/s)}$', [0.2, 1.2])
# fig_va_s.savefig('../../figures/vary_inp/vary_va_pareto-s.png', transparent=True)
# fig_ws_s, _ = hist_scatter_var('ws', r'$w_\mathrm{s}~\mathrm{(wt\%)}$', [10, 30])
# fig_ws_s.savefig('../../figures/vary_inp/vary_ws_pareto-s.png', transparent=True)
# fig_vd_s, _ = hist_scatter_var('vd', r'$V_\mathrm{d}~\mathrm{(\mu L)}$', [0.5, 2.5])
# fig_vd_s.savefig('../../figures/vary_inp/vary_vd_pareto-s.png', transparent=True)
# fig_t_s, _ = hist_scatter_var('t', r'$t~\mathrm{(s)}$', [50, 200])
# fig_t_s.savefig('../../figures/vary_inp/vary_t_pareto-s.png', transparent=True)


# # optuna-mc scatter
# fig_ta_s, _ = hist_scatter_var(
#     optuna_pareto, 'ta', r'$T_\mathrm{a}~\mathrm{(^\circ\hspace{-0.25}C)}$',
#     [61, 74], xticks=[60, 62.5, 65, 67.5, 70, 72.5, 75],
#     xticklabels=[60, 62.5, 65, 67.5, 70, 72.5, 75])
# fig_ta_s.savefig('../../figures/vary_inp/vary_ta_pareto-s-optuna.png', transparent=True)
# fig_va_s, _ = hist_scatter_var(
#     optuna_pareto, 'va', r'$v_\mathrm{a}~\mathrm{(m/s)}$',
#     [0.6, 1.2], xticks=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
#     xticklabels=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
# fig_va_s.savefig('../../figures/vary_inp/vary_va_pareto-s-optuna.png', transparent=True)
# fig_ws_s, _ = hist_scatter_var(
#     optuna_pareto, 'ws', r'$w_\mathrm{s}~\mathrm{(wt\%)}$', [10, 20])
# fig_ws_s.savefig('../../figures/vary_inp/vary_ws_pareto-s-optuna.png', transparent=True)
# fig_vd_s, _ = hist_scatter_var(
#     optuna_pareto, 'vd', r'$V_\mathrm{d}~\mathrm{(\mu L)}$',
#     [0.5, 1.0], xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     xticklabels=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# fig_vd_s.savefig('../../figures/vary_inp/vary_vd_pareto-s-optuna.png', transparent=True)
# fig_t_s, _ = hist_scatter_var(
#     optuna_pareto, 't', r'$t~\mathrm{(s)}$',
#     [120, 200], xticks=[120, 140, 160, 180, 200],
#     xticklabels=[120, 140, 160, 180, 200])
# fig_t_s.savefig('../../figures/vary_inp/vary_t_pareto-s-optuna.png', transparent=True)
