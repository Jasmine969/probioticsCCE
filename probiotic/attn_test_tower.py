import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm, colors
from probiotic.MyNN import ConvAttn2Tower
from torch.nn.utils import rnn
from sklearn.metrics import r2_score
import my_functions as mf
import seaborn as sns

font_formula = fm.FontProperties(
    math_fontfamily='cm', size=19
)
font_text = {'family': 'Times New Roman', 'size': 17}
font_title = {'family': 'Times New Roman', 'size': 20}
font_tick = {'family': 'Times New Roman', 'size': 13}
font_legend = {'family': 'Times New Roman', 'size': 15}
with open('pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
    dct = pickle.load(pf)
    ft_s_tv = dct['ft_s_tv']
    ft_s_test = dct['ft_s_test']
    s_scalar = dct['s_scalar']
model_path = 'trained_models/vali2/tower/lgs_drop589-tsb1'
# model_path = 'optuna_res/test35/vali1/tower/drop398944'
human_itv = True

score_tv, score_test = [], []
model = ConvAttn2Tower(
    in_channels=3, kernel_size=3,
    dk=116, dv=61,
    num_heads=2,
    dropout=0.589,
    act_name='leakyrelu'
).cuda()
model.load_state_dict(torch.load(model_path + '/second_model.pth'))
model.eval()
ticks_dict = {
    121: 30,
    401: 80,
    271: 60,
    301: 60
}


def visualize(
        ft_s, axes,
        itv: bool,
        # title, text_x, text_y,
        plot_scale: str,  # 'lg'/'ori'/'normal'
        ss_max, c, r=None,
        ss_min=None, score_list=None
):
    ft, s, real_ind, length = pack_test(ft_s)
    with torch.no_grad():
        ft = rnn.pad_sequence(ft, batch_first=True).cuda()
        pred, _, score = model(ft, length)
    if score_list is not None:
        score_list.append(score.cpu().squeeze().numpy())
    pred = pred[0, :length[0], [0]]  # normal
    plt.subplots_adjust(hspace=0.338)
    if plot_scale == 'ori':
        pred = pred.cpu().numpy()
    else:
        if plot_scale == 'lg':
            s = np.log10(s)
            pred = torch.log10(pred)
        if itv:
            pred = mf.human_intervene(
                pred.unsqueeze(0).cuda(), ss_max,
                torch.from_numpy(real_ind), False, ss_min
            ).squeeze(0).cpu().numpy()
        elif plot_scale == 'normal':
            pred = torch.clamp(pred, 1e-7, 1).squeeze(0).cpu().numpy()
        else:
            pred = pred.squeeze(0).cpu().numpy()
    t = np.arange(length[0])
    ax = axes[c] if r is None else axes[r, c]
    # ax.scatter(
    #     t[real_ind], s[real_ind],
    #     marker='o', facecolor='white', color='g', label='real'
    # )
    # ax.scatter(
    #     t[real_ind], pred[real_ind], marker='v', label='pred points'
    # )
    # ax.plot(t, pred, label='pred curve')
    ax.scatter(
        t[real_ind], s[real_ind],
        marker='o', facecolor='white', s=90,
        color='r', label=r'ground truth $s^\mathrm{grd}$'
    )
    ax.scatter(
        t[real_ind], pred[real_ind], s=90,
        marker='v', label=r'predicted result of points $\hat{s}^{\mathrm{grd}}$'
    )
    ax.plot(t, pred, '+', markersize=3,
            label=r'predicted result of interpolated labels $\hat{s}^{\mathrm{itp}}$')
    ax.set_xlabel('Time (s)', fontdict=font_text)
    if plot_scale == 'ori':
        y_label = r'$s$'
    elif plot_scale == 'normal':
        y_label = r'$s$'
    else:
        y_label = r'$\lg{s}$'
    ax.set_ylabel(y_label, fontproperties=font_formula)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # set tick labels
    # xticks = ax.get_xticks()
    ticks = np.arange(0, length[0] + 1, ticks_dict[length[0]])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.astype(int), fontdict=font_tick)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks.round(2), fontdict=font_tick)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    try:
        r2 = r2_score(s[real_ind], pred[real_ind])
        ax.text(0.23, 0.08, f'$R^2={r2:.4f}$',
                fontproperties=font_formula, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
    except ValueError:
        ax.set_title(f'lg(negative)')
    # ax.legend(loc='lower left')
    return score_list


def pack_test(ft_s):
    ft = [ft_s[:, :3]]
    s = ft_s[:, [3]].numpy()
    real_ind = ft_s[:, -1].numpy().astype(bool)
    length = [len(ft_s)]
    return ft, s, real_ind, length


def score_heatmap(data, ax, title):
    norm = colors.LogNorm()
    mask = ~np.tril(np.ones_like(data)).astype(bool)
    with sns.axes_style('white'):
        ax = sns.heatmap(
            data, ax=ax, norm=norm, vmax=.3,
            square=True, mask=mask,
            xticklabels=ticks_dict[data.shape[0]],
            yticklabels=ticks_dict[data.shape[0]]
        )
    ax.set_xlabel('Key', fontdict=font_text)
    ax.set_ylabel('Query', fontdict=font_text)
    # set tick labels
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), fontdict=font_tick)
    ax.set_yticklabels(xticks.astype(int), fontdict=font_tick)
    ax.set_title(title, fontdict=font_title, y=-0.45)


# train
fig11, axes11 = plt.subplots(2, 3)
plt.subplots_adjust(
    hspace=0.338, wspace=0.238, left=0.086
)
fig12, axes12 = plt.subplots(2, 3)
plt.subplots_adjust(
    hspace=0.338, wspace=0.238, left=0.086
)
fig13, axes13 = plt.subplots(2, 3)
plt.subplots_adjust(
    hspace=0.338, wspace=0.238, left=0.086
)
for i in range(6):
    row, col = divmod(i, 3)
    score_tv = visualize(
        ft_s_tv[i], axes11, score_list=score_tv,
        ss_max=0, ss_min=None, itv=human_itv,
        plot_scale='lg', r=row, c=col
    )
    _ = visualize(
        ft_s_tv[i], axes12,
        ss_max=1, ss_min=1e-7, itv=human_itv,
        plot_scale='normal', r=row, c=col
    )
    _ = visualize(
        ft_s_tv[i], axes13,
        ss_max=None, ss_min=None, itv=False,
        plot_scale='ori', r=row, c=col
    )
# test
fig31, axes31 = plt.subplots(1, 2)
fig31.set_size_inches(10, 4.5)
fig32, axes32 = plt.subplots(1, 2)
fig32.set_size_inches(10, 4.5)
fig33, axes33 = plt.subplots(1, 2)
fig33.set_size_inches(10, 4.5)
for i in range(2):
    score_test = visualize(
        ft_s_test[i], axes31,
        ss_max=0, ss_min=None, itv=human_itv, score_list=score_test,
        plot_scale='lg', c=i
    )
    _ = visualize(
        ft_s_test[i], axes32,
        ss_max=1, ss_min=1e-7, itv=human_itv,
        plot_scale='normal', c=i
    )
    _ = visualize(
        ft_s_test[i], axes33,
        ss_max=None, ss_min=None, itv=False,
        plot_scale='ori', c=i
    )

# heatmap
# old version, plot all heatmaps
# _, ax5 = plt.subplots(2, 4)
# plt.subplots_adjust(wspace=0.4,hspace=0.3)
# for i in range(6):
#     r, c = divmod(i, 4)
#     score_heatmap(score_tv[i], ax5[r, c],
#                   f'({lower_letter[i]}) Group {xtick_labels_tv[i]} (train+vali)')
# for i in range(2):
#     r, c = 1, 2 + i
#     score_heatmap(score_test[i], ax5[r, c],
#                   f'({lower_letter[i+6]}) Group {xtick_labels_test[i]} (test)')

# new version, plot one heatmap
# fig5, ax5 = plt.subplots()
# score_heatmap(score_test[1], ax5,
#               f'Group {xtick_labels_test[1]} (test)')
# fig5.savefig('figures/heatmap_g5.png', transparent=True)  # 设置整体透明

plt.show()
