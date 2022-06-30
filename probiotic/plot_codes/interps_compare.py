import numpy as np
from scipy import interpolate as itp
from matplotlib import pyplot as plt, font_manager as fm
from functools import partial

font_x = fm.FontProperties(size=20, math_fontfamily='cm')
font_tick = fm.FontProperties(size=13, family='Times New Roman')
font_y = fm.FontProperties(size=20, family='Times New Roman')
s = np.array([1, 0.8, 0.6, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.8, 0.7, 0.3, 0.5, 0.7, 0.05, 0.2, 0.05, 0.1])
t = np.array([0, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450])
tt = np.arange(0, 451)
tt1 = np.r_[np.arange(0, 295), np.linspace(294, 300, 40),
            np.linspace(300,330,200),np.linspace(330,335,40),np.arange(336, 451)]
plt.plot(t, 1 - s, '.', markersize=20, label='real')
plt.xlabel(r'$\mathrm{Time}\ (\mathrm{s})$', fontproperties=font_x)
plt.ylabel('survival rate', fontproperties=font_y)

itps = {
    'PCHIP': itp.PchipInterpolator,
    'linear': partial(itp.interp1d, kind='linear'),
    'quadratic': partial(itp.interp1d, kind='cubic'),
    'nearest': partial(itp.interp1d, kind='nearest'),
    'B-spline': partial(itp.splrep, s=0)
}
styles = ['-.', '-', '-', ':']
for (itp_name, itp_method), style in zip(itps.items(), styles):
    interp = itp_method(t, s)
    if itp_name == 'B-spline':
        ss = itp.splev(tt, interp)
    else:
        ss = interp(tt)
    plt.plot(tt, 1 - ss, label=itp_name, linestyle=style)
sin_lin = itp.interp1d(t, s, kind='linear')
ss1 = sin_lin(tt1)
ad1 = np.sin(30 * np.linspace(0, np.pi * 2, tt1.size)) / 17
ad1[np.isin(tt1, t)] = 0
ss1 = ss1 + ad1
plt.plot(tt1, 1 - ss1, label='linear+sine', linestyle='--', lw=0.8)
plt.xlim([295, 335])
plt.ylim([0.2, 0.6])
plt.xticks(fontproperties=font_tick)
plt.yticks(fontproperties=font_tick)
plt.legend(loc='lower left', prop={'family': 'Times New Roman', 'size': 13})
