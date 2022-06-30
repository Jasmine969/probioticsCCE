from matplotlib import pyplot as plt, font_manager as fm
import pandas as pd
import numpy as np
from brokenaxes import brokenaxes

data = pd.read_excel('model_comparison.xlsx', usecols='A,D:G').dropna().to_numpy()
x = np.arange(14)
plt.rc('font', family='Times New Roman', size=15)
font_formula = fm.FontProperties(
    math_fontfamily='cm', size=20
)
font_text = {'size': 20}
colors = ['#9999ff', '#ff9999', '#b6c9bb', '#f6bc65']
ylim_all = [
    ((-0.95, -0.75), (-0.1, 0.1), (0.55, 1.14)),
    ((-10.8, -10.4), (-9.8, -9.6), (-7.8, -7.4),
     (-3.9, -3.6), (-2.4, -2.2), (-0.18, 1.14)),
    ((0.0, 0.28), (0.5, 1.14)),
    ((-0.64, -0.5), (-0.15, 1.14))
]
bar_width = 0.4


def bax_comp(ind):
    bax = brokenaxes(
        ylims=ylim_all[ind],
        hspace=0.05,
        # wspace=0.05,
        # despine=False,
        d=0.007,
        diag_color='red', tilt=45
    )
    bax.bar(x, data[:, ind + 1], facecolor=colors[ind], width=bar_width)
    bax.set_xticks(x)
    bax.set_xlabel('Model No.', labelpad=28, fontdict=font_text)
    bax.set_ylabel(r'$R^2$', fontproperties=font_formula)
    text_skip = 0.03
    for i in range(14):
        cur_r2 = data[i, ind + 1]
        if cur_r2 > 0:
            bax.text(x[i], cur_r2 + text_skip, f'{cur_r2:.4}', ha='center')
        elif cur_r2 == 0:
            bax.text(x[i], cur_r2 + text_skip, 'Divergence' if i == 8 else 'Unfitted', ha='center')
        else:
            bax.text(x[i], cur_r2 - text_skip, f'{cur_r2:.4}', ha='center', va='top')
    fig = bax.fig
    fig.set_size_inches([15.36, 7.57])
    fig.savefig(f'../figures/model-comp/broken{ind + 1}.jpg')


if __name__ == '__main__':
    # for i in range(4):
    #     bax_comp(i)
    #     plt.close()
    bax_comp(1)