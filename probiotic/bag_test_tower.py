import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm, colors
from torch.nn.utils import rnn
from sklearn.metrics import r2_score
import probioticsCCE.my_functions as mf
import seaborn as sns
from string import ascii_lowercase as lower_letter
from gen_bag_model import gen_bag
from probioticsCCE.my_functions import visualize
from functools import partial


def score_heatmap(data, ax, title=None):
    norm = colors.LogNorm()
    mask = ~np.tril(np.ones_like(data)).astype(bool)
    with sns.axes_style('white'):
        ax = sns.heatmap(
            data, ax=ax, norm=norm, vmax=.3,
            square=True, mask=mask,
            xticklabels=ticks_dict[data.shape[0]],
            yticklabels=ticks_dict[data.shape[0]],
            cbar=False  # specify the colorbar at Line 28
        )
    ax.set_xlabel('Key', fontdict=font_text)
    ax.set_ylabel('Query', fontdict=font_text, labelpad=5)
    ax.figure.colorbar(ax.collections[0])
    # set tick labels
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), fontdict=font_tick)
    ax.set_yticklabels(xticks.astype(int), fontdict=font_tick)
    if title:
        ax.set_title(title, fontdict=font_title, y=-0.45)


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=13)
    font_formula = fm.FontProperties(
        math_fontfamily='cm', size=19
    )
    font_text = {'size': 17}
    font_title = {'size': 20}
    font_tick = {'size': 13}
    font_legend = {'size': 17, 'math_fontfamily': 'cm'}
    with open('pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
        dct = pickle.load(pf)
        ft_s_tv = dct['ft_s_tv']
        ft_s_test = dct['ft_s_test']
        s_scalar = dct['s_scalar']
    human_itv = True
    score_tv, score_test = [], []
    n_train = ['1', '2', '4', '6', '7', '8']
    n_test = ['3', '5']
    ticks_dict = {
        121: 30,
        401: 80,
        271: 60,
        301: 60
    }
    bag_model = gen_bag(ops='win')
    visualize = partial(visualize, model=bag_model)
    # train
    train_ax_adjust = dict(
        left=0.07, right=0.943,
        top=0.921, bottom=0.088,
        hspace=0.338, wspace=0.298
    )

    fig11, axes11 = plt.subplots(2, 3)
    fig11.set_size_inches([15.36, 7.57])
    plt.subplots_adjust(**train_ax_adjust)
    fig12, axes12 = plt.subplots(2, 3)
    fig12.set_size_inches([15.36, 7.57])
    plt.subplots_adjust(**train_ax_adjust)
    # ax
    # texts_x = (
    #     [20, 20, 20, 10, 20, 20],  # tv_lg
    #     [20, 160, 20, 10, 200, 20],  # tv_nm
    #     [15, 15],  # test_lg
    #     [15, 160],  # test_nm
    # )
    # texts_y = (
    #     [-0.4, -2.4, -0.3, -1.7, -1.0, -0.38],  # tv_lg
    #     [0.6, 0.6, 0.65, 0.5, 0.7, 0.6],  # tv_nm
    #     [-1.5, -1.7],  # test_lg
    #     [0.5, 0.5],  # test_nm
    # )
    # fig
    texts_x, texts_y = 0.2, 0.1
    # legend_pos = (
    #     ['lower left'] * 6,
    #     ['lower left', 'center right', 'lower left',
    #      'lower left', 'upper right', 'lower left'],
    #     ['lower left'] * 2,
    #     ['lower left', 'upper right']
    # )
    # ori
    # fig13, axes13 = plt.subplots(2, 3)
    # plt.subplots_adjust(hspace=0.338)
    for i in range(6):
        row, col = divmod(i, 3)
        title = 'G' + n_train[i]
        score_tv = visualize(
            ft_s_tv[i], axes11,
            score_list=score_tv,
            ss_max=0, ss_min=None, itv=human_itv,
            title=title,
            plot_scale='lg', r=row, c=col,
            text_x=texts_x, text_y=texts_y
        )
        _ = visualize(
            ft_s_tv[i], axes12,
            ss_max=1, ss_min=1e-7, itv=human_itv,
            title=title,
            plot_scale='normal', r=row, c=col,
            text_x=texts_x, text_y=texts_y
        )
        # ori
        # _ = visualize(
        #     ft_s_tv[i], axes13,
        #     ss_max=None, ss_min=None, itv=False,
        #     plot_scale='ori', r=row, c=col
        # )

    # legend在PPT添加吧
    # handles, labels = fig11.axes[-1].get_legend_handles_labels()
    # legend_params = {'handles': handles, 'labels': labels,
    #                  'loc': 'lower center', 'ncol': 3, 'prop': font_legend}
    # fig11.legend(**legend_params)
    # fig12.legend(**legend_params)

    fig11.savefig('figures/bagtower/tv-lg.png')
    fig12.savefig('figures/bagtower/tv-lnr.png')

    # heatmap
    # old version, plot all heatmaps
    # xtick_labels_tv = ['1', '2', '4', '6', '7', '8']
    # xtick_labels_test = ['3', '5']
    # _, ax5 = plt.subplots(2, 4)
    # plt.subplots_adjust(wspace=0.4, hspace=0.3)
    # for i in range(6):
    #     r, c = divmod(i, 4)
    #     score_heatmap(score_tv[i], ax5[r, c],
    #                   f'({lower_letter[i]}) Group {xtick_labels_tv[i]} (train+vali)')
    # for i in range(2):
    #     r, c = 1, 2 + i
    #     score_heatmap(score_test[i], ax5[r, c],
    #                   f'({lower_letter[i + 6]}) Group {xtick_labels_test[i]} (test)')

    # new version, plot one heatmap
    # fig5, ax5 = plt.subplots()
    # score_heatmap(score_test[1], ax5)
    # plt.tight_layout()
    # fig6, ax6 = plt.subplots()
    # score_heatmap(score_tv[1], ax6)
    # plt.tight_layout()
    # fig5.savefig('figures/heatmap-g5.jpg')
    # fig6.savefig('figures/heatmap-g2.jpg')

    plt.show()
