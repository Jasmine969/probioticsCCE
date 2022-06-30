import random
from random import uniform
from calc_X_s import calc_x_s
import pickle
import numpy as np
from tqdm import tqdm

random.seed(623)
n = 10000


def run_monte_carlo():
    exp_cond = {
        'Ta': uniform(60, 90) + 273.15,
        'va': uniform(0.2, 1.2),
        'w_rsm': uniform(0.1, 0.3),
        'vd': uniform(0.5, 2.5) * 1e-9,
        'dur': uniform(50, 200)
    }
    x_t, s_t = calc_x_s(
        exp_cond,
        ops='linux'
    )
    cur_res = list(exp_cond.values())
    cur_res[0] -= 273.15
    cur_res[3] *= 1e9
    return np.array([x_t, s_t] + cur_res)


res = run_monte_carlo()
for i in tqdm(range(n - 1)):
    res = np.vstack((res, run_monte_carlo()))
with open('../pickle_data/monte-carlo-dry-ta90-dur200.pkl', 'wb') as pickle_file1:
    pickle.dump(res, pickle_file1)
