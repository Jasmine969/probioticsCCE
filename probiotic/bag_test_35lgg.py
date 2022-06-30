import pickle
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm, colors
import seaborn as sns
from gen_bag_model import gen_bag
from my_functions import visualize
from functools import partial


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
    # ax.set_title(title, fontdict=font_title, y=-0.45)


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=13)
    font_formula = fm.FontProperties(
        math_fontfamily='cm', size=19
    )
    font_text = {'size': 17}
    font_tick = {'size': 13}
    font_legend = {'size': 15}
    human_itv = True
    with open('pickle_data/test35_lgg_s.pkl', 'rb') as pf:
        ft_s_test = pickle.load(pf)
    score_tv, score_test = [], []
    n_test = ['G3', 'G5', 'G9 (LGG)']
    ticks_dict = {
        121: 30, 401: 80,
        271: 60, 301: 60
    }
    bag_model = gen_bag(ops='win')
    visualize = partial(visualize, model=bag_model)
    # ax-text
    # texts_x = (
    #     [15, 160, 220],  # test_nm
    #     [15, 15, 15],  # test_lg
    # )
    # texts_y = (
    #     [0.5, 0.5, 0.7],  # test_nm
    #     [-1.5, -1.7, -0.4],  # test_lg
    # )
    # fig-text
    texts_x, texts_y = 0.2, 0.1
    test_ax_adjust = dict(
        top=0.888, bottom=0.182,
        left=0.053, right=0.987,
        hspace=0.2, wspace=0.295
    )
    fig1, axes1 = plt.subplots(1, 3)
    plt.subplots_adjust(**test_ax_adjust)
    fig1.set_size_inches([15.36, 8/2])
    fig2, axes2 = plt.subplots(1, 3)
    plt.subplots_adjust(**test_ax_adjust)
    fig2.set_size_inches([15.36, 8/2])
    for i in range(3):
        title = n_test[i]
        _ = visualize(
            ft_s_test[i], axes1,
            ss_max=1, ss_min=1e-7, itv=human_itv,
            title=title,
            plot_scale='normal', r=None, c=i,
            text_x=texts_x, text_y=texts_y
        )
        score_test = visualize(
            ft_s_test[i], axes2,
            ss_max=0, ss_min=None, itv=human_itv,
            score_list=score_test, title=title,
            plot_scale='lg', r=None, c=i,
            text_x=texts_x, text_y=texts_y
        )
    fig1.savefig('figures/bagtower/test-lnr.png')
    fig2.savefig('figures/bagtower/test-lg.png')

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
    # score_heatmap(score_test[1], ax5,
    #               f'Group {xtick_labels_test[1]} (test)')
    # fig5.savefig('figures/heatmap_g5.png', transparent=True)  # 设置整体透明

    plt.show()
