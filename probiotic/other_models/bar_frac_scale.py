from matplotlib import pyplot as plt, font_manager as fm
import pandas as pd
import numpy as np
from matplotlib.ticker import NullFormatter, FixedLocator


def forward(x):
    # x = (base_pos ** x - 1) * (x >= 0) + (base_neg ** x - 1) * (x < 0)
    x = 1 / (frac_b - x)
    return x


def inverse(x):
    # x = np.log(x + 1) / np.log(base_pos) * (x >= 0) + np.log(x + 1) / np.log(base_neg) * (x < 0)
    x = frac_b - 1 / x
    return x


data = pd.read_excel('model_comparison.xlsx', usecols='A,D:G').dropna().to_numpy()
x = np.arange(14)
plt.rc('font', family='Times New Roman', size=15)
font_formula = fm.FontProperties(
    math_fontfamily='cm', size=20
)
font_text = {'size': 20}
yticks = (
    [-1.0, -0.4, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [-11, -2.0, -0.5, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [-1.0, -0.4, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
colors = ['#9999ff', '#ff9999', '#b6c9bb', '#f6bc65']
ylims = (
    [-1.4, 1.02],
    [-2000, 1.01],
    [0, 1.02],
    [-1.0, 1.02]
)

bar_width = 0.4
frac_b = 1.6
base_pos, base_neg = 5, 1.1
text_skip = 0.03


def ax_comp(ind):
    ax.bar(x, data[:, ind + 1], facecolor=colors[ind], width=bar_width)
    ax.set_xticks(x)
    ax.set_xlabel('Model No.', labelpad=18, fontdict=font_text)
    ax.set_ylabel(r'$R^2$', fontproperties=font_formula)
    ax.set_yscale('function', functions=(forward, inverse))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(yticks[ind]))
    ax.set_ylim(ylims[ind])
    for i in range(14):
        cur_r2 = data[i, ind + 1]
        # cur_skip = frac_b - cur_r2 - 1 / (text_skip + 1 / (frac_b - cur_r2))
        cur_skip = text_skip
        if cur_r2 > 0:
            ax.text(x[i], cur_r2 + cur_skip, f'{cur_r2:.4}', ha='center')
        elif cur_r2 == 0:
            ax.text(x[i], cur_r2 + cur_skip, 'Divergence' if i == 8 else 'Unfitted', ha='center')
        else:
            ax.text(x[i], cur_r2 - cur_skip, f'{cur_r2:.4}', ha='center', va='top')
    fig.set_size_inches([15.36, 7.57])
    fig.savefig(f'../figures/model-comp/bar-frac{ind + 1}.jpg')


if __name__ == '__main__':
    for i in range(1,2):
        fig, ax = plt.subplots(tight_layout=True)
        ax_comp(i)
