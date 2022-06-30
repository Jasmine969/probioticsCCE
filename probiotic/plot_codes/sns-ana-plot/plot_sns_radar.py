import pickle
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

with open('../../pickle_data/rate5x5x7x5.pkl', 'rb') as pickle_file:
    exp_cond, rates = pickle.load(pickle_file)
rates = np.asarray(rates)
rates_normal = (rates - rates.min()) / rates.ptp()
cmap = get_cmap('RdYlGn', rates.size)
colors = cmap(rates_normal)
angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
angles = np.hstack((angles, [0]))
ta_tuple, va_tuple, ws_tuple, vd_tuple = zip(*exp_cond)
exp_cond_ar = np.array([ta_tuple, va_tuple, ws_tuple, vd_tuple])
exp_cond_ar = (exp_cond_ar - exp_cond_ar.min(axis=1, keepdims=True)
               ) / exp_cond_ar.ptp(axis=1, keepdims=True)
exp_cond_ar = np.vstack((exp_cond_ar, exp_cond_ar[0]))

radar_labels = [r'$T_\mathrm{a}$', r'$v_\mathrm{a}$',
                r'$w_\mathrm{s}$', r'$V_\mathrm{d}$']
fig = plt.figure(facecolor="white", figsize=(10, 6))
plt.subplot(111, polar=True)
for ind, color in enumerate(colors):
    plt.plot(angles, exp_cond_ar[:,ind] + 0.25, 'o-', linewidth=1.5, alpha=0.2, color=color)
# plt.plot(angles, exp_cond_ar[:,[100]] + 0.25, 'o-', linewidth=1.5, alpha=0.2)
plt.thetagrids(angles[:-1] * 180 / np.pi,
               radar_labels)
sm = plt.cm.ScalarMappable(cmap)
sm.set_array([])
plt.colorbar(sm)