from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import font_manager as fm
import pickle

# data = pd.read_excel('../vary_inp_test/optuna_res_dry/opt1.xlsx',
#                      usecols='B,C,F:J').dropna().to_numpy()
with open('../../pickle_data/monte-carlo-dry-ta90-dur200.pkl', 'rb') as pf1:
    data = pickle.load(pf1)
# data = data[data[:, 0] <= 1]  # select those X < 1 kg/kg
colors = cm.get_cmap('coolwarm')

x, s, var = np.split(data, [1, 2], axis=-1)
x = x.ravel()
s = s.ravel()
# var = (var - var.min(axis=0)) / var.ptp(axis=0)
ta, va, ws, vd, t = (each.ravel() for each in np.split(var, [1, 2, 3, 4], axis=-1))
plt.rc('font', family='Times New Roman', size=25)
font_formula = dict(math_fontfamily='cm', size=29)
font_text = {'size': 21}


def scatter_corr_var(var, vmin, vmax, cb_label):
    # scatter and calc corr
    fig, ax = plt.subplots(figsize=(12.1, 9))
    sc = ax.scatter(x, s, c=var, cmap=colors,
                    vmin=vmin, vmax=vmax,
                    alpha=0.7)
    # ax.set_xlabel(r'$X~(\mathrm{kg/kg})$', fontdict=font_formula)
    # ax.set_ylabel(r'$s$', fontdict=dict(math_fontfamily='cm', size=28))
    cb = plt.colorbar(sc)
    cb.set_label(cb_label, fontdict=font_formula)
    corr_var = {
        'moisture content': np.corrcoef(var, x)[0, 1],
        'survival': np.corrcoef(var, s)[0, 1]
    }
    corr_var['double'] = (abs(corr_var['survival']) + abs(corr_var['moisture content'])) * 0.5
    plt.close(fig)
    return fig, ax, corr_var, cb


# scatter ta and corr
ta_ticks = list(range(65, 115, 5))
fig_ta, ax_ta, corr_ta, cb_ta = scatter_corr_var(
    ta, 60, 90,
    r'$T_\mathrm{a}~(\mathrm{^\circ\hspace{-0.25}C})$'
)
fig_ta.savefig('../../figures/vary_inp/ta_scatter.png', transparent=True)
ax_ta.set_xlim([0, 1])
ax_ta.set_ylim([0, 1])
cb_ta.remove()
fig_ta.savefig('../../figures/vary_inp/ta_scatter_enlarge.png', transparent=True)
# scatter va
fig_va, ax_va, corr_va, cb_va = scatter_corr_var(
    va, 0.2, 1.2,
    r'$v_\mathrm{a}~(\mathrm{m/s})$'
)
fig_va.savefig('../../figures/vary_inp/va_scatter.png', transparent=True)
ax_va.set_xlim([0, 1])
ax_va.set_ylim([0, 1])
cb_va.remove()
fig_va.savefig('../../figures/vary_inp/va_scatter_enlarge.png', transparent=True)
# scatter ws
fig_ws, ax_ws, corr_ws, cb_ws = scatter_corr_var(
    ws * 100, 10, 30,
    r'$w_\mathrm{s}~(\mathrm{wt\%})$'
)
fig_ws.savefig('../../figures/vary_inp/ws_scatter.png', transparent=True)
ax_ws.set_xlim([0, 1])
ax_ws.set_ylim([0, 1])
cb_ws.remove()
fig_ws.savefig('../../figures/vary_inp/ws_scatter_enlarge.png', transparent=True)
# scatter vd
fig_vd, ax_vd, corr_vd, cb_vd = scatter_corr_var(
    vd, 0.5, 2.5,
    r'$V_\mathrm{d}~(\mathrm{\mu L})$'
)
fig_vd.savefig('../../figures/vary_inp/vd_scatter.png', transparent=True)
ax_vd.set_xlim([0, 1])
ax_vd.set_ylim([0, 1])
cb_vd.remove()
fig_vd.savefig('../../figures/vary_inp/vd_scatter_enlarge.png', transparent=True)
# scatter t
fig_t, ax_t, corr_t, cb_t = scatter_corr_var(
    t, 50, 200,
    r'$t~(\mathrm{s})$'
)
fig_t.savefig('../../figures/vary_inp/t_scatter.png', transparent=True)
ax_t.set_xlim([0, 1])
ax_t.set_ylim([0, 1])
cb_t.remove()
fig_t.savefig('../../figures/vary_inp/t_scatter_enlarge.png', transparent=True)
colors = ['#9999ff', '#ff9999', '#b6c9bb']
bar_width = 0.4
fig_bar = plt.figure(6, figsize=[7.7, 15])
plt.subplots_adjust(top=1)
bar_skip = 0.005
y_ori = np.arange(1, 8, 1.5)
y_name = [r'$T_\mathrm{a}$',
          r'$v_\mathrm{a}$',
          r'$w_\mathrm{s}$',
          r'$V_\mathrm{d}$',
          r'$t$']
legend_labels = [r'$|\rho_{\theta,X}|$', r'$|\rho_{\theta,s}|$', r'$(|\rho_{\theta,X}|+|\rho_{\theta,s}|)/2$']
for ind, corr_name in enumerate(corr_vd.keys()):
    y_data = y_ori + (ind - 1) * 0.4
    x_data = [corr_ta[corr_name], corr_va[corr_name],
              corr_ws[corr_name], corr_vd[corr_name], corr_t[corr_name]]
    plt.barh(y_data, x_data,
             height=bar_width, facecolor=colors[ind], label=legend_labels[ind])
    for xx, yy in zip(x_data, y_data):
        plt.text(xx + bar_skip, yy, f'{xx:.2f}', va='center')
    plt.yticks(y_ori, y_name, fontproperties=fm.FontProperties(**font_formula))
# plt.xlim([0, 0.73])
# plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.45), prop=font_formula)
plt.xlabel('Absolute correlation coefficient', fontdict=font_text)
plt.ylabel('Drying condition')
# fig_bar.savefig('../figures/vary_inp/bar_corr.png', transparent=True)
