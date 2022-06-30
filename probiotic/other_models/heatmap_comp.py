import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager as fm, colors


def forward(x):
    # x = (base_pos ** x - 1) * (x >= 0) + (base_neg ** x - 1) * (x < 0)
    x = 1 / (frac_b - x)
    return x


def inverse(x):
    # x = np.log(x + 1) / np.log(base_pos) * (x >= 0) + np.log(x + 1) / np.log(base_neg) * (x < 0)
    x = frac_b - 1 / x
    return x


def comp_heatmap(ax):
    plt.rc('font', family='Times New Roman', size=15)
    plt.subplots_adjust(left=0.05, right=1)
    norm = colors.FuncNorm(
        (forward, inverse),
        vmin=-11, vmax=1
    )
    mask = np.zeros_like(data)
    mask[:, [8]] = 1
    mask = mask.astype(np.bool)
    with sns.axes_style('white'):
        ax = sns.heatmap(
            data, ax=ax, vmax=.3,
            mask=mask,
            annot=True, fmt='.4g',
            annot_kws=font_annot,
            norm=norm,
            xticklabels=np.arange(13),
            yticklabels=np.arange(4),
            cbar=False,
            cmap='RdYlGn'
        )
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([-11, -1.0, 0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # set tick labels
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks.astype(int), **font_tick)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels(['', '', '', ''])
    return ax


font_formula = fm.FontProperties(
    math_fontfamily='cm', size=22
)
font_text = {'size': 22, 'fontfamily': 'Times New Roman'}
font_annot = {'size': 17, 'fontfamily': 'Times New Roman'}
font_tick = {'size': 18, 'fontfamily': 'Times New Roman'}
fig, axes = plt.subplots()
data = pd.read_excel('model_comparison.xlsx', usecols='D:G').dropna().to_numpy()
data = data.T
base_pos, base_neg = 5, 1.1
frac_b = 1.5
ax = comp_heatmap(axes)
fig.set_size_inches([15.36, 7.57])
fig.savefig('../figures/model-comp/heatmap.jpg')
