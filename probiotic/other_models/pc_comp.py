import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.path import Path
import matplotlib.patches as patches

data = pd.read_excel('model_comparison.xlsx', usecols='A:B,D:G,J').dropna()
category = data['category'].to_list()
colorval = 100 - data['colorval'].to_numpy()
cate_dict = {'CSAN': 3, 'Kinetic': 2, 'Statistical': 1}
cate_num = list(map(lambda x: cate_dict[x], category))
ys = data.iloc[:, [0, 2, 3, 4, 5]].to_numpy()
ys = np.insert(ys, 1, values=cate_num, axis=1)
ys[ys < 0] = 0
N = ys.shape[0]
plt.rc('font', family='Times New Roman', size=18)
font_formula = dict(math_fontfamily='cm', size=23)
font_text = {'size': 23}
ynames = ['Model No.', 'Category',
          r'$R^2_\mathrm{fit,nm}$',
          r'$R^2_\mathrm{fit,lg}$',
          r'$R^2_\mathrm{test,nm}$',
          r'$R^2_\mathrm{test,lg}$']

fig, host = plt.subplots()
fig.set_size_inches([19.2, 9.83])

# organize the data
ymins = np.min(ys, axis=0)
ymaxs = np.max(ys, axis=0)
ymins[2:], ymaxs[2:] = 0, 1
ymins[0], ymaxs[0] = ymaxs[0], ymins[0]  # reverse the model axis
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
        if i == 1:
            ax.yaxis.set_ticks([1, 2, 3])
            ax.yaxis.set_ticklabels(list(reversed(list(cate_dict.keys()))))

host.set_xlim(0, ys.shape[1] - 1)
host.set_xticks(range(ys.shape[1]))
host.set_xticklabels(ynames, fontdict=font_formula)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)
host.xaxis.tick_top()

cmap = get_cmap('rainbow', N)
colors = cmap(colorval / 100)
colors[:, :3] = np.clip(colors[:, :3] / 1.2, 0, 1)
colors[[1, 8, 10, 13], -1] *= 0.17
# colors[-1, :3] = np.clip(colors[-1, :3] / 1.9, 0, 1)
colors[-2, :] = np.array([1, 0, 0, 1])  # 让model2红色变浅
x_control = np.linspace(0, ys.shape[0] - 1, ys.shape[0] * 3 - 2, endpoint=True).tolist()
for j in range(N):
    # to just draw straight lines between the axes:
    # host.plot(range(ys.shape[1]), zs[j, :], c=colors[(cate_num[j] - 1) % len(colors)])

    # create bezier curves
    # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
    #   at one third towards the next axis; the first and last axis have one less control vertex
    # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
    # y-coordinate: repeat every point three times, except the first and last only twice
    if j in [1, 8, 10, 13]:
        y_control = np.repeat(zs[j, :2], 3)[1:-1]
    else:
        y_control = np.repeat(zs[j, :], 3)[1:-1]
    verts = list(zip(x_control, y_control))
    # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1 if j else 2, edgecolor=colors[j])
    host.add_patch(patch)
# for j in range(2):
#     verts = list(zip(x_control, np.repeat(zs_unfitted[j, :], 3)[1:-1]))
#     codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
#     path = Path(verts, codes)
#     patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors_unfitted[j])
#     host.add_patch(patch)
plt.tight_layout()
plt.show()
fig.savefig('../figures/model_comp_pc.pdf')
