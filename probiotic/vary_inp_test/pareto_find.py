from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import font_manager as fm
from matplotlib.ticker import AutoMinorLocator
import pickle
from my_functions import is_pareto_efficient_simple
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist


def x2ws(x):
    return 1 / (x + 1) * 100


with open('../pickle_data/monte-carlo-dry-ta90-dur200.pkl', 'rb') as pf:
    data_mc = pickle.load(pf)

# mode = 'zh'  # chinese
mode = 'eng'
# data_mc = data_mc[data_mc[:, 0] <= 1 / 0.7 - 1]  # ws>=70%
data_mc = data_mc[data_mc[:, 1] >= 0]  # s>=0.1
data_mc[:, 4] *= 100
plt.rc('font', family='Times New Roman', size=24)
font_formula = dict(math_fontfamily='cm', size=30)
font_legend = dict(family='SimHei', size=24)
font_text = {'size': 34}
font_text_zh = {'size': 31, 'family': 'SimHei'}

pure_mc_xs = data_mc[:, [0, 1]]
pure_mc_xs[:, 1] *= -1
is_pareto_mc = is_pareto_efficient_simple(pure_mc_xs)
data_mc[:, 0] = x2ws(data_mc[:, 0])
mc_pareto = data_mc[is_pareto_mc, :]
non_pareto = data_mc[~is_pareto_mc, :]
fig_pareto = plt.figure(figsize=(9, 7.4))
plt.scatter(non_pareto[:, 0], non_pareto[:, 1],
            c='#ff7f0e', alpha=0.3, s=40, marker='s')
plt.scatter(mc_pareto[:, 0], mc_pareto[:, 1], s=40,
            c='#1f77b4', marker='s')
plt.xlabel(r'$w_{\mathrm{s}}~(\mathrm{wt\%})$', fontdict=font_formula)
plt.ylabel(r'$s$', fontdict=font_formula)
# plt.gcf().savefig('../figures/vary_inp/pareto-mc.png', transparent=True)
plt.close(fig_pareto)

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


markersize = 15
fig_num = 1


def hist_var(data_pareto, var_name, xlabel, bins,
             x_minor_num, xticks, yticks=None,
             y_minor_num=2, log=False):
    global fig_num
    plt.figure(fig_num, figsize=(15, 7))
    fig_num += 1
    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(left=0.07, right=0.85, top=0.976, bottom=0.143)
    par_X = host.twinx()
    par_s = host.twinx()
    par_s.axis["right"] = par_s.new_fixed_axis(loc="right", offset=(80, 0))
    par_X.axis["right"].toggle(all=True)
    par_s.axis["right"].toggle(all=True)

    # fig_, host = plt.subplots(figsize=(14, 6.28))
    host.grid(axis='y', which='major', alpha=0.5)
    x_minor_locator = AutoMinorLocator(x_minor_num)
    host.xaxis.set_minor_locator(x_minor_locator)
    y_minor_locator = AutoMinorLocator(y_minor_num)
    host.yaxis.set_minor_locator(y_minor_locator)
    host.hist(
        [each[:, var2col[var_name]] for each in data_pareto],
        bins=bins, log=log,
        alpha=0.3, edgecolor='k',
        stacked=False
    )
    # host.set_xlabel(xlabel, fontdict=font_formula)
    # if mode == 'zh':
    #     host.set_ylabel('频数', fontdict=font_text_zh)
    # else:
    #     host.set_ylabel('Count', fontdict=font_text)
    host.set_xticks(xticks)
    if yticks is not None:
        host.set_yticks(yticks)
    host.spines['top'].set_visible(False)

    df_pareto = pd.DataFrame(data_pareto[0])
    df_pareto['pos'] = pd.cut(df_pareto.iloc[:, var2col[var_name]], bins)
    mean_pareto = df_pareto.groupby(by='pos').mean().dropna()
    x_mean_pareto = [each.mid for each in mean_pareto.index]
    y_mean_pareto = mean_pareto.iloc[:, :2].to_numpy()
    df_non = pd.DataFrame(data_pareto[1])
    df_non['pos'] = pd.cut(df_non.iloc[:, var2col[var_name]], bins)
    mean_non = df_non.groupby(by='pos').mean().dropna()
    x_mean_non = [each.mid for each in mean_non.index]
    y_mean_non = mean_non.iloc[:, :2].to_numpy()
    par_X.plot(x_mean_pareto, y_mean_pareto[:, 0], '*-', color='C0', markersize=markersize)
    par_X.plot(x_mean_non, y_mean_non[:, 0], '*-', color='C1', markersize=markersize)
    par_s.plot(x_mean_pareto, y_mean_pareto[:, 1], '>--', color='C0', markersize=markersize)
    par_s.plot(x_mean_non, y_mean_non[:, 1], '>--', color='C1', markersize=markersize)
    return plt.gcf(), plt.gca()


# mc hist
fig_ta_op, _ = hist_var([mc_pareto, non_pareto], 'ta', r'$T_\mathrm{a}~\mathrm{(^\circ\hspace{-0.25}C)}$',
                        x_minor_num=5, bins=list(range(60, 110, 2)),
                        xticks=list(range(60, 111, 10)), log=True,
                        y_minor_num=5)
# fig_ta_op.savefig('../figures/vary_inp/vary_ta_pareto-op.png', transparent=True)
# fig_va_op, _ = hist_var([mc_pareto, non_pareto], 'va', r'$v_\mathrm{a}~\mathrm{(m/s)}$',
#                         x_minor_num=3, bins=np.arange(0.45, 1.05, 0.05),
#                         xticks=[0.45, 0.6, 0.75, 0.9, 1.05], log=True,
#                         y_minor_num=5)
# fig_va_op.savefig('../figures/vary_inp/vary_va_pareto-op.png', transparent=True)
# fig_ws_op, _ = hist_var([mc_pareto, non_pareto], 'ws', r'$w_\mathrm{s}~\mathrm{(wt\%)}$',
#                         x_minor_num=4, bins=np.arange(5.5, 20.1, 0.5),
#                         xticks=[4, 6, 8, 10, 12, 14, 16, 18, 20], log=True
#                         )
# fig_ws_op.savefig('../figures/vary_inp/vary_ws_pareto-op.png', transparent=True)
# fig_vd_op, _ = hist_var([mc_pareto, non_pareto], 'vd', r'$V_\mathrm{d}~\mathrm{(\mu L)}$',
#                         x_minor_num=5, bins=np.arange(0.5, 2.5, 0.1),
#                         xticks=np.arange(0.5, 2.51, 0.5), log=True
#                         )
# fig_vd_op.savefig('../figures/vary_inp/vary_vd_pareto-op.png', transparent=True)
# fig_t_op, _ = hist_var([mc_pareto, non_pareto], 't', r'$t~\mathrm{(s)}$',
#                        x_minor_num=5, bins=np.arange(50, 300, 10),
#                        xticks=list(range(50, 301, 50))
#                        )
# fig_t_op.savefig('../figures/vary_inp/vary_t_pareto-op.png', transparent=True)


# # mc-mc scatter
# fig_ta_s, _ = hist_scatter_var(
#     mc_pareto, 'ta', r'$T_\mathrm{a}~\mathrm{(^\circ\hspace{-0.25}C)}$',
#     [61, 74], xticks=[60, 62.5, 65, 67.5, 70, 72.5, 75],
#     xticklabels=[60, 62.5, 65, 67.5, 70, 72.5, 75])
# fig_ta_s.savefig('../../figures/vary_inp/vary_ta_pareto-s-mc.png', transparent=True)
# fig_va_s, _ = hist_scatter_var(
#     mc_pareto, 'va', r'$v_\mathrm{a}~\mathrm{(m/s)}$',
#     [0.6, 1.2], xticks=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
#     xticklabels=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
# fig_va_s.savefig('../../figures/vary_inp/vary_va_pareto-s-mc.png', transparent=True)
# fig_ws_s, _ = hist_scatter_var(
#     mc_pareto, 'ws', r'$w_\mathrm{s}~\mathrm{(wt\%)}$', [10, 20])
# fig_ws_s.savefig('../../figures/vary_inp/vary_ws_pareto-s-mc.png', transparent=True)
# fig_vd_s, _ = hist_scatter_var(
#     mc_pareto, 'vd', r'$V_\mathrm{d}~\mathrm{(\mu L)}$',
#     [0.5, 1.0], xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#     xticklabels=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# fig_vd_s.savefig('../../figures/vary_inp/vary_vd_pareto-s-mc.png', transparent=True)
# fig_t_s, _ = hist_scatter_var(
#     mc_pareto, 't', r'$t~\mathrm{(s)}$',
#     [120, 200], xticks=[120, 140, 160, 180, 200],
#     xticklabels=[120, 140, 160, 180, 200])
# fig_t_s.savefig('../../figures/vary_inp/vary_t_pareto-s-mc.png', transparent=True)

plt.show()
