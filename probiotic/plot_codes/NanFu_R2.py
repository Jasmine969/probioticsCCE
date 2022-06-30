import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import font_manager as fm


def pred(temp):
    kd = np.exp(46.8627732) * np.exp(-1.44034755e5 / 8.314 / temp[1:])
    s_hat = 1 / np.cumprod(kd + 1)
    s_hat = np.hstack(([1], s_hat))
    return s_hat


symbol = ['gs', 'rv', 'ko']  # 70 90 110
plt.rc('font', family='Times New Roman', size=17)
fig1, ax1 = plt.subplots()
fig1.set_size_inches([7.7, 6.1])
fig2, ax2 = plt.subplots()
fig2.set_size_inches([7.7, 6.1])
axes = [ax1, ax2]
font_formula = fm.FontProperties(math_fontfamily='cm', size=23)
font_text = {'size': 23}
for i in range(6):
    r, c = i // 3, i % 3
    df = pd.read_excel('excel/itp_ft_s.xlsx', sheet_name='Sheet' + str(i + 1))
    data = df[['t(s)', 'T(K)', 'itp']].to_numpy()
    tag = df['tag'].to_numpy()
    s_hat = np.log10(pred(data[:, 1]))
    s_real = np.log10(data[tag, 2])
    r2 = r2_score(s_real, s_hat[tag])
    axes[r].plot(data[:, 0], s_hat, symbol[c], markerfacecolor='w')
    axes[r].plot(data[tag, 0], s_real, symbol[c], markeredgecolor='k')
    axes[r].set_xlabel('Time (s)', fontdict=font_text)
    axes[r].set_ylabel(r'$\lg s$', fontproperties=font_formula)
    axes[r].set_xlim([0, 320])
    print(i, r2)
fig1.savefig('figures/fu-R2-10wt.jpg')
fig2.savefig('figures/fu-R2-20wt.jpg')
plt.show()
