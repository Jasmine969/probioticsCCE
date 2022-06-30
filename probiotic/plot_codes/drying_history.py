from matplotlib import pyplot as plt, font_manager as fm
import numpy as np
import functools as ftool

font_formula = fm.FontProperties(
    math_fontfamily='cm',
    size=28
)
font_tick = fm.FontProperties(
    family='Times New Roman', size=20
)
font_text = fm.FontProperties(
    family='Times New Roman', size=28
)
n = 501
n1 = int((n - 1) / 5 + 1)
t = np.linspace(0, 10, n)
T1 = 19.15 + 51.64 / (1+ 59.38 * np.exp(-0.817 * t))
T2 = 70 - 50 * np.exp(-0.557 * t)
T3 = np.full(n, 70.)
T3[:n1] = t[:n1] * 25 + 20
fig = plt.figure(figsize=(8.39, 6.26))
plot = ftool.partial(plt.plot, lw=3)
plot(t, T1)
plot(t, T2)
plot(t, T3)
plt.xlabel('Time  (s)', fontproperties=font_text, labelpad=16)
plt.ylabel(r'$T_\mathrm{d}\quad (^\circ\mathrm{C})$', fontproperties=font_formula, labelpad=6)
plt.xticks(np.arange(0, 11, 2), fontproperties=font_tick)
plt.yticks(np.arange(20, 71, 10), fontproperties=font_tick)
plt.tight_layout()
fig.savefig('E:/tex_code/tex_probiotics/figure-jpg/diff_history.jpg', transparent=True)  # 设置整体透明
plt.show()
