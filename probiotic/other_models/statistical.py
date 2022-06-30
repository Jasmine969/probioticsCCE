import numpy as np
from sklearn.metrics import r2_score
import pickle
from scipy.optimize import curve_fit

with open('../pickle_data/for_other_models_dataset.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    s_gt = dct['s_gt']
    s_test = dct['s_test']
    t_rst = dct['t_rst']
    t_test = dct['t_test']


def sigmoidal(x, alpha, omega, sigma, tau):
    return alpha + (omega - alpha) / (1 + np.exp(4 * sigma * (tau - x) / (omega - alpha)))


p_0 = [0, -10, 1, 2.4]
p_opts = []
for i in range(6):
    t_rs = t_rst[i]
    t_rs[0] = 0.1
    t_rs = np.log10(t_rs)
    s = np.log10(s_gt[i])
    p_opt, _ = curve_fit(sigmoidal, t_rs, s, p_0, bounds=(
        [-np.inf, -np.inf, -np.inf, -np.inf], [0.23, -0.5, np.inf, np.inf]))
    p_opts.append(p_opt)
