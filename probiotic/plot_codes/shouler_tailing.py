from matplotlib import pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from scipy.interpolate import pchip
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from functools import partial


def sigmoidal(x, alpha, omega, sigma, tau):
    return alpha + (omega - alpha) / (
            1 + np.exp(4 * sigma * (tau - x) / (omega - alpha)))


t = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240])
tt = np.linspace(0, max(t), 500)
y = np.array([1, 1, 0.95, 0.7, 0.02, 0.01, 0.008, 0.007, 0.0068])
# 预测的应该是一条曲线，先给出散点，再插值成曲线
# popt, _ = curve_fit(
#     sigmoidal, t, lg_y_hat,
#     p0=[1, -4, -0.01, 110],
#     bounds=[(-0.9, -5, -0.04, 110), (2, -3, 0, 120)]
# )
# print(f'ori_r2: {r2_score(lg_y_hat, sigmoidal(t, *popt)):.4f}')
# print(popt)
# interp = pchip(t, lg_y_hat)
# lg_yy_hat = interp(tt)
popt = [0.87, -2, -0.04, 121.98]

sigmoidal = partial(sigmoidal, alpha=popt[0], omega=popt[1], sigma=popt[2], tau=popt[3])
lg_yy_hat = sigmoidal(tt)
# 某一秒后弄成直线，临界点保持相切
cut_off_ind = 280
cut_off = tt[cut_off_ind]
slope = (sigmoidal(cut_off) - sigmoidal(cut_off - 0.01)) / 0.01
lg_yy_hat[cut_off_ind:] = slope * (tt[cut_off_ind:] - cut_off) + sigmoidal(cut_off)
lg_yy_hat -= 0.87
lg_y = np.log10(y)
yy_hat = 10 ** lg_yy_hat
plt.rc('font', family='Times New Roman', size=16)
font_formula = fm.FontProperties(
    math_fontfamily='cm', size=21
)
font_legend = {
    'size': 19,
    'family': 'SimHei'  # 中文
}
font_text = {'size': 21}
size = 80
# linear plot
fig1, ax1 = plt.subplots(tight_layout=True)
plt.scatter(
    t, y,
    # label='experimental',
    label='实验值',  # 中文
    marker='o', facecolor='white', color='r', s=size)
plt.plot(
    tt, yy_hat,
    # label='predicted'
    label='预测值'  # 中文
)
plt.legend(loc='best', prop=font_legend)
plt.xlabel('Time (s)', fontdict=font_text)
plt.ylabel(r'$s$', fontproperties=font_formula)
# log plot
fig2, ax2 = plt.subplots(tight_layout=True)
plt.scatter(
    t, lg_y,
    # label='experimental',
    label='实验值',  # 中文
    marker='o', facecolor='white', color='r', s=size
)
plt.plot(
    tt, lg_yy_hat,
    # label='predicted'
    label='预测值'  # 中文
)

plt.legend(loc='best', prop=font_legend)
plt.xlabel('Time (s)', fontdict=font_text)
plt.ylabel(r'$\lg{s}$', fontproperties=font_formula)

# save
fig1.savefig('../figures/shoulder-tailing/线性.png')
fig2.savefig('../figures/shoulder-tailing/对数.png')
