import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

font_label = fm.FontProperties(
    math_fontfamily='cm',
    size=30
)
font_tick = {
    'family': 'Times New Roman', 'size': 16
}
font_text = {
    'size': 21, 'math_fontfamily': 'cm'
}
s = np.linspace(0, 1, 200)
w_lg = 1.008 / (1 + 0.009428 * np.exp(10.78 * s))
fig, ax = plt.subplots(figsize=(6, 4.5))
plt.subplots_adjust(
    top=0.967, bottom=0.192, left=0.155,
    right=0.97, hspace=0.2, wspace=0.2
)
plt.plot(s, w_lg, lw=4)
cp = 1 / np.log(10)
plt.plot(cp, 0.5, 'r.', markersize=20)
plt.text(cp + 0.04, 0.47,
         r'$(\dfrac{1}{\ln{10}},\dfrac{1}{2})$',
         fontdict=font_text)
plt.plot(0, 1, 'r.', markersize=20, label=r'hhh')
plt.text(-0.04, 0.9, r'$(0,1)$', fontdict=font_text)

plt.plot(1, 0, 'r.', markersize=20)
plt.text(0.90, 0.06, r'$(1,0)$', fontdict=font_text)
plt.xlabel(r'$\bar{s}$', fontproperties=font_label)
plt.ylabel(r'$w_\mathrm{lg}$', fontproperties=font_label)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels(xticks.round(2), fontdict=font_tick)
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels(yticks.round(2), fontdict=font_tick)
plt.xlim(xlim)
plt.ylim(ylim)
plt.tight_layout()
fig.savefig('../figures/w_lg.svg', transparent=True)  # 设置整体透明
plt.show()
