import random
from calc_X_s import calc_x_s
import pickle
import numpy as np
from tqdm import tqdm
from itertools import product
from collections import OrderedDict

random.seed(623)
# sample number of vary var
n_vary = 10
# sample number of fixed var
n_fixed = 5
ub_lb = {
    'Ta': [60 + 273.15, 100 + 273.15],
    'va': [0.2, 1.2],
    'ws': [0.1, 0.3],
    'vd': [0.5e-9, 2.5e-9],
    'dur': [50, 300]
}


def run_sns_ana(var_name):
    exp_cond = OrderedDict()
    for key, val in ub_lb.items():
        exp_cond[key] = np.linspace(
            val[0], val[1], n_vary if key == var_name else n_fixed)
    exp_cond_ar = np.asarray(
        list(product(*list(exp_cond.values())))
    )
    xs_ar = np.zeros((exp_cond_ar.shape[0], 2))
    for ind in tqdm(range(exp_cond_ar.shape[0])):
        x, s = calc_x_s(
            dict(zip(ub_lb.keys(), exp_cond_ar[ind, :].tolist())),
            ops='linux'
        )
        xs_ar[ind, :] = np.array([x, s])
    res_ar = np.hstack((xs_ar, exp_cond_ar))
    res_ar[:, 2] -= 273.15
    res_ar[:, 5] *= 1e9
    return res_ar


res_dct = {}
for key in ub_lb.keys():
    print(f'\nbegin {key}\n')
    res_dct[key] = run_sns_ana(key)
with open('../pickle_data/sns-ana-dry.pkl', 'wb') as pickle_file1:
    pickle.dump(res_dct, pickle_file1)
