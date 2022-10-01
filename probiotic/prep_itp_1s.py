from scipy.interpolate import PchipInterpolator as phicp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm
import my_functions as mf

writer = pd.ExcelWriter('excel\\itp_ft_s.xlsx')
font_title = fm.FontProperties(size=18, family='Times New Roman')
font_xy = fm.FontProperties(size=14, family='Times New Roman')
font_tick = {'fontsize': 13, 'family': 'Times New Roman'}

for i in range(8):
    df_ft_c = pd.read_excel('excel\\raw_ft_s.xlsx', sheet_name='Sheet' + str(i + 1))
    t = df_ft_c['t(s)'].to_numpy()
    tt_sep = [np.arange(t[i], t[i + 1]) for i in range(t.size - 1)]
    tt_sep.append(np.array([t[-1]]))
    s = df_ft_c['s'].to_numpy()
    s = mf.mono_decrease(s)
    df_1s = pd.read_excel('excel\\raw_1s.xlsx', sheet_name='Sheet' + str(i + 1))
    interp = phicp(t, s)
    ss_sep = [interp(each) for each in tt_sep]
    df_1s['itp'] = np.hstack(ss_sep)
    df_1s['tag'] = df_1s['t(s)'].isin(df_ft_c['t(s)'])
    # visualize(ax1 if i < 4 else ax2, t, s,
    #           df_1s['t(s)'].to_numpy(), df_1s['itp'].to_numpy(),
    #           df_1s['s_ub'].to_numpy(), df_1s['s_lb'].to_numpy(), pos=(i // 2, i % 2))
    df_1s.to_excel(writer, index=None, sheet_name='Sheet' + str(i + 1))
    df_1s['ws'] = df_1s['X'].apply(lambda x: 1 / (x + 1) * 100)
    try:
        print(f"G{i+1}: {df_1s[df_1s['ws'] > 90].iloc[0, -3]}")
    except Exception:
        print(f'G{i+1}: None')
writer.save()
plt.show()
