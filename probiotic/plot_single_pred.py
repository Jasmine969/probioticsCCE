import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm
from MyNN import ConvAttention, BaggingModel, CoupledModel
from torch.nn.utils import rnn
from sklearn.metrics import r2_score
import probioticsCCE.my_functions as mf
from copy import deepcopy as dc

font_formula = fm.FontProperties(
    math_fontfamily='cm', size=20
)
font_text = {'family': 'Times New Roman', 'size': 16}
font_title = {'family': 'Times New Roman', 'size': 18}
font_tick = {'family': 'Times New Roman', 'size': 13}
font_legend = {'family': 'Times New Roman', 'size': 14}
size_obs = 35
size_pred = 65
lw_pred = 2
title_y = -0.34
with open('pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
    dct = pickle.load(pf)
    ft_s_test = dct['ft_s_test']
    ft_s_tv = dct['ft_s_tv']
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
coupled_model = CoupledModel(
    lg_net=bag_lg_model, normal_net=bag_normal_model
)

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


def visualize(
        ft_s, axes, model,
        text_x, text_y,
        lg_ori: bool, c,
        plot_scale: str,  # 'lg'/'ori'/'normal'
        itv=True,
        r=None, legend_pos='best',
        title=None, log_scale=False
):
    ft, s, real_ind, length = pack_test(ft_s)
    with torch.no_grad():
        ft = rnn.pad_sequence(ft, batch_first=True).cuda()
        if isinstance(model, CoupledModel):
            _, _, pred, _ = model(ft, length)
        else:
            pred, _ = model(ft, length)
    pred = pred[0, :length[0], [0]]
    plt.subplots_adjust(hspace=0.338)
    if lg_ori:
        ss_max, ss_min = 0, None
    else:
        ss_max, ss_min = 1, 1e-6
    if itv:
        pred = mf.human_intervene(
            pred.unsqueeze(0).cuda(), ss_max,
            torch.from_numpy(real_ind), ss_min
        ).squeeze(0).cpu().numpy()
    else:
        ori_pred = pred.cpu().numpy()[-10:]
        pred = mf.human_intervene(
            pred.unsqueeze(0).cuda(), ss_max,
            torch.from_numpy(real_ind), ss_min
        ).squeeze(0).cpu().numpy()
    if plot_scale == 'normal':
        if lg_ori:
            pred = 10 ** pred
    elif plot_scale in ['lg', 'log']:
        s = np.log10(s)
        if not lg_ori:
            pred = np.log10(pred)
    t = np.arange(length[0])
    ax = axes[c] if r is None else axes[r, c]
    ax.scatter(
        t[real_ind], s[real_ind],
        marker='o', facecolor='white', color='g',
        label='observed points', s=size_obs
    )
    ax.scatter(
        t[real_ind], pred[real_ind], marker='v',
        # label='predicted (at observed t.p.)',
        label='predicted points',
        s=size_pred, color='C0'
    )
    ax.plot(t, pred, linewidth=lw_pred,
            label='predicted curve')
    # ax.scatter(t[~real_ind], pred[~real_ind],
    # marker='s', s=5, facecolor='C0',
    # label='predicted (at interpolated t.p.'
    # )
    ax.set_xlabel('Time (s)', fontdict=font_text)
    if plot_scale == 'normal':
        y_label = r'$s$'
    else:
        y_label = r'$\lg{s}$'
    ax.set_ylabel(y_label, fontproperties=font_formula)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # set tick labels
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
        ax.text(text_x, text_y, f'$R^2={r2:.4f}$', fontproperties=font_formula)
    except ValueError:
        ax.text(text_x, text_y, f'lg(negative)')
    if bool(title):
        ax.set_title(title, fontproperties=font_title, y=title_y)
    # ax.set_xlim([107.5, 121])
    # xtick = ax.get_xticks()
    # ax.set_xticks(xtick)
    # ax.set_xticklabels(list(map(int,xtick)),fontdict=font_tick)
    # if log_scale:
    #     ax.set_ylim([1e-7, 0.014])
    #     ax.set_yscale('log')
    #     ax.plot(t[:-9], np.ones_like(t[:-9]) * 1e-6, 'r--')
    #     yticks = ax.get_yticks()
    #     ax.set_yscale('log')
    #     ax.set_yticks(yticks)
    #     ax.set_yticklabels([f'{ytick:.0E}' for ytick in yticks],fontdict=font_tick)
    #     ax.set_xlim([107.5, 121])
    # else:
    #     ax.set_ylim([-0.05, 0.014])
    #     ytick = ax.get_yticks()
    #     ax.set_yticks(ytick)
    #     ax.set_yticklabels(ytick.round(2),fontdict=font_tick)
    #     ax.set_ylim([-0.05, 0.014])
    #     ax.scatter(np.arange(120, 121), ori_pred[-1],
    #                marker='v', facecolor='w',edgecolor='C0',
    #                s=50, label='predicted (original, at observed t.p.)')
    #     ax.scatter(np.arange(111, 120), ori_pred[:-1],
    #                marker='s',facecolor='w',edgecolor='C0',
    #                s=50, label='predicted (original, at interpolated t.p.)')
    #     ax.plot(t[:-9], np.ones_like(t[:-9]) * 1e-6, 'r--')
    #     ax.fill_between(
    #         np.linspace(100, 130), np.ones(50) * -0.05,
    #         np.zeros(50), color='#f75545', alpha=0.3)
    #     ax.set_xlim([107.5, 121])
    legend = ax.legend(loc=legend_pos, prop=font_legend, frameon=False)
    # legend = plt.legend(loc='lower right', prop=font_legend, frameon=False)
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')  # 设置图例legend背景透明
    # plt.tight_layout()


ticks_dict = {
    121: 30,
    401: 80,
    271: 60,
    301: 60
}


def pack_test(ft_s):
    ft = [ft_s[:, :3]]
    s = ft_s[:, [3]].numpy()
    real_ind = ft_s[:, -1].numpy().astype(bool)
    length = [len(ft_s)]
    return ft, s, real_ind, length


fig31, axes31 = plt.subplots(2, 2)
fig31.set_size_inches(11.78, 7)

visualize(
    ft_s_test[0], axes31, coupled_model,
    title='(a) Group 3 on the normal scale',
    lg_ori=False,
    plot_scale='normal',
    r=0,
    c=0, text_x=15, text_y=0.6, legend_pos='lower left'
)

visualize(
    ft_s_test[0], axes31, coupled_model,
    title='(b) Group 3 on the log scale',
    lg_ori=False,
    plot_scale='log',
    r=0,
    c=1, text_x=15, text_y=-1.3, legend_pos='lower left'
)
visualize(
    ft_s_test[1], axes31, coupled_model,
    title='(c) Group 5 on the normal scale',
    lg_ori=False,
    plot_scale='normal',
    r=1,
    c=0, text_x=170, text_y=0.3, legend_pos='upper right'
)

visualize(
    ft_s_test[1], axes31, coupled_model,
    title='(d) Group 5 on the log scale',
    lg_ori=False,
    plot_scale='log',
    r=1,
    c=1, text_x=15, text_y=-1.5, legend_pos='lower left'
)
plt.subplots_adjust(
    top=0.955, bottom=0.13, left=0.07,
    right=0.93, hspace=0.41, wspace=0.245
)
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
fig31.savefig('figures/cpt_test_res.png', transparent=True)
# visualize(
#     ft_s_tv[3], axes31, bag_normal_model,
#     title='(b) Prediction of normal model on the log scale',
#     lg_ori=False,
#     plot_scale='log', r=0, c=1,
#     text_x=15, text_y=-2, legend_pos='lower left'
# )
#
# visualize(
#     ft_s_tv[3], axes31, bag_lg_model,
#     title='(b) Prediction of log model on the log scale',
#     lg_ori=True,
#     plot_scale='log', r=1, c=1,
#     text_x=15, text_y=-1.8, legend_pos='lower left'
# )

# ==============
# plot tv6 non-itv normal on normal
# fig2, axes2 = plt.subplots(1, 2)
# visualize(
#     ft_s_tv[3], axes2, bag_normal_model,
#     lg_ori=False,
#     plot_scale='normal', c=0, itv=False,
#     text_x=50, text_y=0.4, legend_pos='lower left'
# )
# visualize(
#     ft_s_tv[3], axes2, bag_normal_model,
#     lg_ori=False,
#     plot_scale='normal', c=1,
#     text_x=50, text_y=0.4, legend_pos='best',
#     log_scale=True
# )
plt.show()
