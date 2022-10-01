import random
from random import uniform
from calc_X_s import calc_x_s
import pickle
import numpy as np
from tqdm import tqdm

random.seed(623)
n = 30000


def run_monte_carlo():
    x_t, s_t = 20, -0.01
    exp_cond = {}
    while s_t < 0:
        exp_cond = {
            'Ta': uniform(50, 110) + 273.15,
            'va': uniform(0.1, 1.5),
            'ws': uniform(0.05, 0.3),
            'vd': uniform(0.5, 3) * 1e-9,
            'dur': uniform(50, 300)
        }
        x_t, s_t = calc_x_s(exp_cond)
    cur_res = list(exp_cond.values())
    cur_res[0] -= 273.15
    cur_res[3] *= 1e9
    return np.array([x_t, s_t] + cur_res)


res = run_monte_carlo()
for i in tqdm(range(n - 1)):
    res = np.vstack((res, run_monte_carlo()))
with open('../../pickle_data/monte-carlo-dry-ta90-dur200.pkl', 'wb') as pickle_file1:
    pickle.dump(res, pickle_file1)
