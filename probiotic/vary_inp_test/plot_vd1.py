from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import ticker

fig, ax = plt.subplots(2, 4)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)  # 科学计数法
labels = ['1', '1.5', '2']
titles = [
    r'$C1$',
    r'$d_p$',
    r'$h_m$',
    r'$-\mathrm{d}m/\mathrm{d}t$',
    r'$A_p$',
    r'$\rho_s$',
    r'$\rho_b$',
    r'$\rho_{s_b}$'
]
for i in range(3):
    data = pd.read_excel(
        'vd1.xlsx', sheet_name='Sheet' + str(i + 1), header=0).to_numpy()
    data = np.c_[np.arange(data.shape[0]), data]
    for j in range(8):
        r, c = j // 4, j % 4
        ax[r, c].plot(data[:, 0], data[:, j + 1], label=labels[i])
for j in range(8):
    r, c = j // 4, j % 4
    ax[r, c].set_ylabel(titles[j], fontdict={'math_fontfamily': 'cm', 'size': 20})
    ax[r, c].legend(loc='best')
# plt.subplots_adjust(left=0.086,right=0.96,wspace=0.305)
