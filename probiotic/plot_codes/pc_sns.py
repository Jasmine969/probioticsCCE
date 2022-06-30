import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def softmax(x):
    tmp = np.log(x)
    return tmp / tmp.sum()


data = pd.read_excel('../vary_inp_test/optuna_res_dry/opt1.xlsx',
                     usecols='B,C,F:J').dropna()
data = data[data.iloc[:, 0] < 0.3]
colors = data['survival rate'].to_numpy()
colors = -1 / colors ** 0.01
colors = (colors - colors.min()) / colors.ptp()
opacities = 1 / (data['X (kg/kg)'].to_numpy()) ** 1.5
opacities = (opacities - opacities.min()) / opacities.ptp()
# opacities = 1 - opacities  # less X means more transparent
ys = data.to_numpy()
N = ys.shape[0]
plt.rc('font', family='Times New Roman', size=18)
font_formula = dict(math_fontfamily='cm', size=23)
font_text = {'size': 23}
ynames = [r'$X~(\mathrm{kg/kg})$', r'$s$',
          r'$T_\mathrm{a}~(\mathrm{^\circ\hspace{-0.25}C})$',
          r'$v_\mathrm{a}~(\mathrm{m/s})$',
          r'$w_\mathrm{s}~(\mathrm{wt\%})$',
          r'$V_\mathrm{d}~(\mathrm{\mu L})$',
          'Time (s)']
fig, host = plt.subplots()
# fig.set_size_inches([19.2, 9.83])

# organize the data
ymins = np.min(ys, axis=0)
ymaxs = np.max(ys, axis=0)
# ymins[0], ymaxs[0] = ymaxs[0], ymins[0]  # reverse the model axis
dys = ymaxs - ymins
ymins -= dys * 0.05  # add 5% padding below and above
ymaxs += dys * 0.05
dys = ymaxs - ymins

# transform all data to be compatible with the main axis (min-max-map)
zs = np.zeros_like(ys)
zs[:, 0] = ys[:, 0]
zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
zs_unfitted = np.array([[1., 6.], [10., 12.]])

axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

host.set_xlim(0, ys.shape[1] - 1)
host.set_xticks(range(ys.shape[1]))
host.set_xticklabels(ynames, fontdict=font_formula)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)
host.xaxis.tick_top()

cmap = get_cmap('RdYlGn', N)
colors = cmap(colors)
colors[:, -1] = opacities
# colors[:, :3] = np.clip(colors[:, :3] / 1.2, 0, 1)
# colors[[1, 8, 10, 13], -1] *= 0.17
# # colors[-1, :3] = np.clip(colors[-1, :3] / 1.9, 0, 1)
# colors[-2, :] = np.array([1, 0, 0, 1])  # 让model2红色变浅
# x_control = np.linspace(0, ys.shape[0] - 1, ys.shape[0] * 3 - 2, endpoint=True).tolist()
for j in range(N):
    # to just draw straight lines between the axes:
    host.plot(range(ys.shape[1]), zs[j, :], c=colors[j])

plt.tight_layout()
plt.show()
fig.savefig('../figures/model_comp_pc.pdf')
