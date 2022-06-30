import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm, colors
from MyNN import ConvAttention, CoupledModel, BaggingModel
from torch.nn.utils import rnn
from sklearn.metrics import r2_score
import probioticsCCE.my_functions as mf
from copy import deepcopy as dc
import math
import seaborn as sns
from string import ascii_lowercase as lower_letter


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
parent_path = 'trained_models/test35/no_added_files/'
# load lg models
lg_model1 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=118, dv=55, num_heads=2, dropout=0.256,
    act_name='leakyrelu'
).cuda()
lg_model1_path = \
    parent_path + 'vali1/lgs_non-tilde_mse_pueracc/best_model.pth'
lg_model1.load_state_dict(torch.load(lg_model1_path))

lg_model2 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=128, dv=95, num_heads=1, dropout=0.341,
    act_name='leakyrelu'
).cuda()
lg_model2_path = \
    parent_path + 'vali2/lgs_non-tilde_lw1/second_model.pth'
lg_model2.load_state_dict(torch.load(lg_model2_path))

lg_model3 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=124, dv=138, num_heads=2, dropout=0.325,
    act_name='leakyrelu'
).cuda()
lg_model3_path = \
    parent_path + 'vali3/lgs_non-tilde2/best_model.pth'
lg_model3.load_state_dict(torch.load(lg_model3_path))

lg_model4 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=108, dv=86, num_heads=5, dropout=0.148,
    act_name='leakyrelu'
).cuda()
lg_model4_path = \
    parent_path + 'vali4/lgs_non-tilde/best_model.pth'
lg_model4.load_state_dict(torch.load(lg_model4_path))

lg_model5 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=59, dv=46, num_heads=1, dropout=0.292,
    act_name='leakyrelu'
).cuda()
lg_model5_path = \
    parent_path + 'vali5/lgs_non-tilde/second_model.pth'
lg_model5.load_state_dict(torch.load(lg_model5_path))

lg_model6 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=61, dv=94, num_heads=2, dropout=0.289,
    act_name='leakyrelu'
).cuda()
lg_model6_path = \
    parent_path + 'vali6/lgs_non-tilde/second_model.pth'
lg_model6.load_state_dict(torch.load(lg_model6_path))

# load normal models
normal_model1 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=49, dv=32, num_heads=3, dropout=0.229,
    act_name='leakyrelu'
).cuda()
normal_model1_path = \
    parent_path + 'vali1/s_non-tilde_mse_pueracc_lw0d6/best_model.pth'
normal_model1.load_state_dict(torch.load(normal_model1_path))

normal_model2 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=109, dv=53, num_heads=3, dropout=0.414,
    act_name='leakyrelu'
).cuda()
normal_model2_path = \
    parent_path + 'vali2/s_non-tilde/best_model.pth'
normal_model2.load_state_dict(torch.load(normal_model2_path))

normal_model3 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=126, dv=76, num_heads=3, dropout=0.205,
    act_name='leakyrelu'
).cuda()
normal_model3_path = \
    parent_path + 'vali3/s_non-tilde/best_model.pth'
normal_model3.load_state_dict(torch.load(normal_model3_path))

normal_model4 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=45, dv=92, num_heads=3, dropout=0.332,
    act_name='leakyrelu'
).cuda()
normal_model4_path = \
    parent_path + 'vali4/s_non-tilde/best_model.pth'
normal_model4.load_state_dict(torch.load(normal_model4_path))

normal_model5 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=84, dv=128, num_heads=1, dropout=0.264,
    act_name='leakyrelu'
).cuda()
normal_model5_path = \
    parent_path + 'vali5/s_non-tilde2/second_model.pth'
normal_model5.load_state_dict(torch.load(normal_model5_path))

normal_model6 = ConvAttention(
    in_channels=3, kernel_size=3,
    dk=100, dv=112, num_heads=5, dropout=0.342,
    act_name='leakyrelu'
).cuda()
normal_model6_path = \
    parent_path + 'vali6/s_non-tilde/second_model.pth'
normal_model6.load_state_dict(torch.load(normal_model6_path))

bag_lg_model = BaggingModel([
    lg_model1, lg_model2, lg_model3,
    lg_model4, lg_model5, lg_model6
], [0.9724, 0.9747, 0.9619, 0.9916, 0.9526, 0.9790])
bag_normal_model = BaggingModel([
    normal_model1, normal_model2, normal_model3,
    normal_model4, normal_model5, normal_model6
], [0.9419, 0.9834, 0.9736, 0.9842, 0.9852, 0.9841])
model = CoupledModel(normal_net=bag_normal_model, lg_net=bag_lg_model)
human_itv = True

s_max, s_min = 1, 1e-6
lg_acc_tv = {
    'normal_model': [],
    'lg_model': [],
    'coupled_model': []
}
lg_acc_test = dc(lg_acc_tv)
normal_acc_tv = dc(lg_acc_tv)
normal_acc_test = dc(lg_acc_tv)
score_tv, score_test = [], []
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
        ss_max, acc_dict,
        c, r=None, scalar=None,
        ss_min=None, score_list=None
):
    ft, s, real_ind, length = pack_test(ft_s)
    with torch.no_grad():
        ft = rnn.pad_sequence(ft, batch_first=True).cuda()
        normal_pred, lg_pred, pred, score = model(ft, length)
    if score_list is not None:
        score_list.append(score.cpu().squeeze().numpy())
    pred = pred[0, :length[0], [0]]
    normal_pred = normal_pred[0, :length[0], [0]]
    lg_pred = lg_pred[0, :length[0], [0]]
    plt.subplots_adjust(hspace=0.338)
    if plot_scale == 'ori':
        pred = pred.cpu().numpy()
    else:
        if itv:
            pred = mf.human_intervene(
                pred.unsqueeze(0).cuda(), ss_max,
                torch.from_numpy(real_ind), ss_min
            ).squeeze(0).cpu().numpy()
            normal_pred = mf.human_intervene(
                normal_pred.unsqueeze(0).cuda(), ss_max,
                torch.from_numpy(real_ind), ss_min
            ).squeeze(0).cpu().numpy()
            lg_pred = mf.human_intervene(
                lg_pred.unsqueeze(0).cuda(), ss_max,
                torch.from_numpy(real_ind), ss_min
            ).squeeze(0).cpu().numpy()
        else:
            pred = torch.clamp(pred, 1e-7, 1).squeeze(0).cpu().numpy()
            lg_pred = torch.clamp(lg_pred, 1e-7, 1).squeeze(0).cpu().numpy()
            normal_pred = torch.clamp(normal_pred, 1e-7, 1).squeeze(0).cpu().numpy()
        if plot_scale == 'lg':
            s = np.log10(s)
            pred = np.log10(pred)
            lg_pred = np.log10(lg_pred)
            normal_pred = np.log10(normal_pred)
    t = np.arange(length[0])
    ax = axes[c] if r is None else axes[r, c]
    ax.scatter(
        t[real_ind], s[real_ind],
        marker='o', facecolor='white', color='g', label='real'
    )
    ax.scatter(
        t[real_ind], pred[real_ind], marker='v', label='pred points'
    )
    ax.plot(t, pred, label='pred curve')
    ax.set_xlabel('Time (s)', fontdict=font_text)
    if plot_scale == 'ori':
        y_label = 'ori'
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
        if bool(acc_dict):
            acc_dict['coupled_model'].append(r2)
            acc_dict['normal_model'].append(
                r2_score(s[real_ind], normal_pred[real_ind])
            )
            acc_dict['lg_model'].append(
                r2_score(s[real_ind], lg_pred[real_ind])
            )
        ax.set_title(f'$R^2={r2:.4f}$', fontproperties=font_formula)
    except ValueError:
        ax.set_title(f'lg(negative)')
    ax.legend(loc='lower left')
    return acc_dict, score_list


def pack_test(ft_s):
    ft = [ft_s[:, :3]]
    s = ft_s[:, [3]].numpy()
    real_ind = ft_s[:, -1].numpy().astype(bool)
    length = [len(ft_s)]
    return ft, s, real_ind, length


def bar_group(ax, acc_dict, title, xticklabels,
              legend_pos='best', ylim=None):
    bar_width = 0.2
    params = {
        'width': bar_width, 'align': 'center', 'alpha': 0.5
    }
    x = np.arange(len(acc_dict['lg_model']))
    ax.bar(x, acc_dict['normal_model'],
           color='c', label='normal-output model', **params)
    ax.bar(x + bar_width, acc_dict['lg_model'],
           color='b', label='logarithmic-output model', **params)
    ax.bar(x + bar_width * 2, acc_dict['coupled_model'],
           color='orange', label='coupled model', **params)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=legend_pos, prop=font_legend)
    ax.set_xlabel('Group code', fontdict=font_text)
    ax.set_ylabel('R-squared', fontdict=font_text)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(xticklabels, rotation=0, fontdict=font_text)
    ax.set_title(title, fontdict=font_title, y=-0.4)
    total_acc = acc_dict['normal_model'] + acc_dict['lg_model'] \
                + acc_dict['coupled_model']
    if ylim is None:
        ylim = [min(total_acc) - 0.005, max(total_acc) + 0.005]
    yticks = np.linspace(
        math.floor(ylim[0] * 1000) / 1000,
        math.ceil(ylim[1] * 1000) / 1000, 5
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{ytick:.3f}' for ytick in yticks], fontdict=font_text)
    ax.set_ylim(ylim)


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
plt.subplots_adjust(hspace=0.338)
fig12, axes12 = plt.subplots(2, 3)
plt.subplots_adjust(hspace=0.338)
for i in range(6):
    row, col = divmod(i, 3)
    lg_acc_tv, score_tv = visualize(
        ft_s_tv[i], axes11, acc_dict=lg_acc_tv, score_list=score_tv,
        ss_max=s_max, ss_min=s_min, itv=human_itv,
        plot_scale='lg', scalar=s_scalar, r=row, c=col
    )
    normal_acc_tv, _ = visualize(
        ft_s_tv[i], axes12, acc_dict=normal_acc_tv,
        ss_max=s_max, ss_min=s_min, itv=human_itv,
        plot_scale='normal', scalar=s_scalar, r=row, c=col
    )

# test
fig31, axes31 = plt.subplots(1, 2)
fig32, axes32 = plt.subplots(1, 2)
for i in range(2):
    lg_acc_test, score_test = visualize(
        ft_s_test[i], axes31,
        acc_dict=lg_acc_test,
        ss_max=s_max, ss_min=s_min, itv=human_itv, score_list=score_test,
        plot_scale='lg', scalar=s_scalar, c=i
    )
    normal_acc_test, _ = visualize(
        ft_s_test[i], axes32,
        acc_dict=normal_acc_test,
        ss_max=s_max, ss_min=s_min, itv=human_itv,
        plot_scale='normal', scalar=s_scalar, c=i
    )
# old version bar plot, for train and test
# fig4, axes4 = plt.subplots(2, 2)
# plt.subplots_adjust(hspace=0.36)
#
# legend_position = ['lower left', 'center left', 'lower right', 'center left']
# ylim_dict = [[0.92, 1], [0.4, 1], [0.9, 1], [-0.36, 1]]
# xtick_labels_tv = ['1', '2', '4', '6', '7', '8']
# xtick_labels_test = ['3', '5']
# bar_group(
#     axes4[0, 0], normal_acc_tv, '(a) Train accuracy on the normal scale',
#     xtick_labels_tv, legend_pos=legend_position[0],
#     ylim=ylim_dict[0]
# )
# bar_group(
#     axes4[0, 1], lg_acc_tv, '(b) Train accuracy on the logarithmic scale',
#     xtick_labels_tv, legend_pos=legend_position[1],
#     ylim=ylim_dict[1]
# )
# bar_group(
#     axes4[1, 0], normal_acc_test, '(c) Test accuracy on the normal scale',
#     xtick_labels_test, legend_pos=legend_position[2],
#     ylim=ylim_dict[2]
# )
# bar_group(
#     axes4[1, 1], lg_acc_test, '(d) Test accuracy on the logarithmic scale',
#     xtick_labels_test, legend_pos=legend_position[3],
#     ylim=ylim_dict[3]
# )
# new version bar plot, for test only
fig4, axes4 = plt.subplots(2, 2)
fig4.set_size_inches(14.4, 6.42)

legend_position = ['lower right', 'center left']
ylim_dict = [[0.9, 1], [-0.36, 1]]
xtick_labels_test = ['3', '5']
bar_group(
    axes4[0, 1], normal_acc_test, '(a) Test accuracy on the normal scale',
    xtick_labels_test, legend_pos=legend_position[0],
    ylim=ylim_dict[0]
)
bar_group(
    axes4[1, 1], lg_acc_test, '(b) Test accuracy on the logarithmic scale',
    xtick_labels_test, legend_pos=legend_position[1],
    ylim=ylim_dict[1]
)
plt.subplots_adjust(
    top=0.955, bottom=0.13, left=0.13,
    right=0.888, hspace=0.46, wspace=0.29
)
fig4.savefig('figures/bar_test_acc.pdf', transparent=True)
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
