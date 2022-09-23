from my_functions import least_square
import pickle
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

with open('../pickle_data/vary_cond_t02.pkl', 'rb') as pkl_t02:
    dct = pickle.load(pkl_t02)
    ta = np.asarray(dct['Ta'], dtype=np.float_)
    va = np.asarray(dct['va'], dtype=np.float_)
    ws = np.asarray(dct['ws'], dtype=np.float_)
    vd = np.asarray(dct['vd'], dtype=np.float_)
xx = np.linspace(0, 1)
font_text = {'family': 'Times New Roman', 'size': 21}
font_title = {'family': 'Times New Roman', 'size': 23}
font_formula = {'math_fontfamily': 'cm', 'size': 21}
plt.rc('font', family='Times New Roman', size=18)
scale_fac = 2
plt.figure(1, figsize=(4*scale_fac, 3*scale_fac))
# ta fit
ta_X = np.c_[ta[0, :], np.ones(ta.shape[1])]
ta_y = ta[1, :]
k_ta, b_ta = least_square(ta_X, ta_y)
fit_ta = xx * k_ta + b_ta
plt.plot(xx, fit_ta, color='C0')
plt.scatter(ta_X[:, 0], ta_y, marker='o', color='C0', label=r'$T_\mathrm{a}$')
# va fit
va_X = np.c_[va[0, :], np.ones(va.shape[1])]
va_y = va[1, :]
k_va, b_va = least_square(va_X, va_y)
fit_va = xx * k_va + b_va
plt.plot(xx, fit_va, color='C1')
plt.scatter(va_X[:, 0], va_y, marker='s', color='C1', label=r'$v_\mathrm{a}$')
# ws fit
ws_X = np.c_[ws[0, :], np.ones(ws.shape[1])]
ws_y = ws[1, :]
k_ws, b_ws = least_square(ws_X, ws_y)
fit_ws = xx * k_ws + b_ws
plt.plot(xx, fit_ws, color='C2')
plt.scatter(ws_X[:, 0], ws_y, marker='d', color='C2', label=r'$w_\mathrm{s,0}$')
# vd fit
vd_X = np.c_[vd[0, :], np.ones(vd.shape[1])]
vd_y = vd[1, :]
k_vd, b_vd = least_square(vd_X, vd_y)
fit_vd = xx * k_vd + b_vd
plt.plot(xx, fit_vd, color='C3')
plt.scatter(vd_X[:, 0], vd_y, marker='o', color='C3', label=r'$V_\mathrm{d}$')

print(f'ta slope: {k_ta}')
print(f'va slope: {k_va}')
print(f'ws slope: {k_ws}')
print(f'vd slope: {k_vd}')
plt.legend(loc='best', prop=font_formula)
plt.xlabel('Normalized drying condtions', fontdict=font_text)
plt.ylabel(r'$t_{\hat{s}=0.2}~(\mathrm{s})$', fontdict=font_formula)
plt.gcf().savefig('../figures/vary_inp/ols_reg_cond.png')
plt.show()
pass
