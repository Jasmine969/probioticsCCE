import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt
from functools import partial
from brokenaxes import brokenaxes
from matplotlib import font_manager as fm
from copy import deepcopy as dc
import my_functions as mf


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


plt.rc('font', family='Times New Roman', size=17)
tick_size = 18
font_text = {'size': 23}
font_formula = {'math_fontfamily': 'cm', 'size': 23}
path = '../trained_models/vali2/' \
       'tower/lgs_drop589-tsb1/tensorboard_res/'
loss_name = 'events.out.tfevents.1641619737.LAPTOP-1QA0JPIO.22368.0'
lg_train_name = 'accuracy_lg_train/' \
                'events.out.tfevents.1641619741.LAPTOP-1QA0JPIO.22368.3'
lg_vali_name = 'accuracy_lg_vali/' \
               'events.out.tfevents.1641619741.LAPTOP-1QA0JPIO.22368.4'
lnr_train_name = 'accuracy_normal_train/' \
                 'events.out.tfevents.1641619741.LAPTOP-1QA0JPIO.22368.1'
lnr_vali_name = 'accuracy_normal_vali/' \
                'events.out.tfevents.1641619741.LAPTOP-1QA0JPIO.22368.2'
val_loss = 'loss/epoch'
val_lg = 'accuracy_lg'
val_lnr = 'accuracy_normal'
names = [loss_name, lnr_train_name, lnr_vali_name, lg_train_name, lg_vali_name]
vals = [val_loss, val_lnr, val_lnr, val_lg, val_lg]
res = {'loss': None,
       'lnr_train': None, 'lnr_vali': None,
       'lg_train': None, 'lg_vali': None}
for i, key in enumerate(res.keys()):
    res[key] = read_tensorboard_data(path + names[i], vals[i])
    res[key] = np.asarray([j.value for j in res[key]])
train_ar = res['lnr_train'] * 0.4 + res['lg_train'] * 0.6
vali_ar = res['lnr_vali'] * 0.4 + res['lg_vali'] * 0.6
# plot 1273 epoch
epochs = np.arange(1, 2001)
# # loss plot
# win_size = 51
# loss_smooth = mf.conv_smooth(res['loss'], win_size)

# # loss global
# bax = brokenaxes(
#     xlims=((-10, 2002),),
#     ylims=((0.0059, 0.0572), (0.21, 0.62)),
#     yscale='log',
#     hspace=0.05, wspace=0.05, despine=False, d=0.007,
#     diag_color='red'
# )
# bax.fig.set_size_inches([12.51, 6.82])
# bax.semilogy(epochs, res['loss'], label='original')
# # bax.semilogy(epochs[win_size // 2:-(win_size // 2)],
# #              loss_smooth, label='smooth', lw=2.5)
# # bax.legend(loc='upper right', prop={'size': 20})
# bax.set_xlabel('Epoch', fontdict=font_text, labelpad=30)
# bax.set_ylabel(r'$\mathcal{L}$', fontproperties=fm.FontProperties(
#     math_fontfamily='cm', size=24
# ), labelpad=70)
# # bax.legend(loc='upper right', prop={'size': 20})
# bax.plot(np.linspace(-10, 2002, 500), np.ones(500) * 6.4e-3, 'r--')  # flood
# bax.fig.savefig('../figures/loss-R2/tsb-loss.png', transparent=True)

# # loss-1274
# plt.semilogy(epochs, res['loss'], label='original')
# # plt.semilogy(epochs[win_size // 2:-(win_size // 2)],
# #              loss_smooth, label='smooth', lw=2.5)
# plt.subplots_adjust(top=0.85,left=0.18,right=0.948)
# plt.xlabel(
#     'Epoch', fontdict=font_text,
#     labelpad=-320
# )
# plt.ylabel(r'$\mathcal{L}$', fontproperties=fm.FontProperties(
#     math_fontfamily='cm', size=24
# ))
# plt.xlim([1266.5, 1279.5])
# plt.ylim([0.0075, 0.0105])
# plt.gca().xaxis.tick_top()
# plt.gcf().set_size_inches([7.09, 5.04])
# plt.gcf().savefig('figures/loss-1274.jpg')

# R2 plot
# bax = brokenaxes(xlims=((-10, 300), (750, 1400), (1900, 2002)),
#                  ylims=((-2.83, -2.7), (0.3, 1.01)),
#                  hspace=0.05, wspace=0.05, despine=False, d=0.007,
#                  diag_color='red')
# bax.plot(epochs, res['lnr_train'], 'v-', label='lnr_train', markersize=2)
# bax.plot(epochs, res['lnr_vali'], 's-', label='lnr_vali', markersize=2)
# bax.plot(epochs, res['lg_train'], '>-', label='lg_train', markersize=2)
# bax.plot(epochs, res['lg_vali'], '.-', label='lg_vali', markersize=6)
# bax.set_xlabel('Epoch', fontdict=font_text, labelpad=26)
# bax.set_ylabel(r'$R^2$', fontdict=font_formula, labelpad=36)
# plt.gcf().set_size_inches([15.36, 7.57])
# plt.gcf().savefig('../figures/loss-R2/tsb-total.svg', transparent=True)
# fig, ax = plt.gcf(), plt.gca()
# ax.text(0.32, 0.89, '300', transform=fig.transFigure)
# ax.text(0.36, 0.89, '750', transform=fig.transFigure)
# ax.text(0.78, 0.89, '1400', transform=fig.transFigure)
# ax.text(0.83, 0.89, '1900', transform=fig.transFigure)
# ax.text(0.089, 0.195, r'$-$',
#         transform=fig.transFigure, fontdict={'math_fontfamily': 'cm'})
# ax.text(0.1, 0.19, '2.7', transform=fig.transFigure)
# ax.text(0.1, 0.24, '0.3', transform=fig.transFigure)


plt.plot(epochs, res['lnr_train'], 's-', label='lnr,train', markersize=5)
plt.plot(epochs, res['lnr_vali'], 'v-', label='lnr,vali', markersize=5)
plt.plot(epochs, res['lg_train'], '>-', label='lg,train', markersize=5)
plt.plot(epochs, res['lg_vali'], '.-', label='lg,vali', markersize=6)
plt.gca().xaxis.tick_top()
plt.xlabel(
    'Epoch', fontdict=font_text,
    labelpad=-330  # start used
    # labelpad=-390  # 1274 used
)
plt.ylabel(r'$R^2$', fontdict=font_formula)
plt.gcf().set_size_inches([5.46, 5.19])  # start used
plt.subplots_adjust(left=0.21, right=0.962, top=0.857)  # start used
plt.xlim([0, 9.5])  # start used
plt.ylim([-3, 1])  # start used
plt.legend(loc='lower right', prop={'size': 23})  # start used

# plt.gcf().set_size_inches([10.03, 6.32])  # 1274 used
# plt.subplots_adjust(top=0.843)
# plt.xlim([1267.5, 1279.58])  # 1274 used
# plt.ylim([0.936, 0.996])  # 1274 used

plt.gcf().savefig('../figures/loss-R2/tsb-start.png')
# plt.gcf().savefig('../figures/loss-R2/tsb-1274.png')

# ax.xaxis.tick_top()
# for x in [0, 0.2, .4, .6, .8, 1]:
#     for y in [0, 0.2, .4, .6, .8, 1]:
#         ax.text(x, y, str(x) + ' ' + str(y), transform=fig.transFigure)
# plt.tight_layout()
