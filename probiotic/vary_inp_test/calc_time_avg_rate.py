from probiotic.gen_bag_model import gen_bag
import numpy as np
import my_functions as mf
import pickle
import torch
from torch.nn.utils import rnn
from probiotic.drying_kinetics import zhuh_REA_SDD as ReaSdd
from scipy.integrate import simps
from itertools import product
from tqdm import tqdm


def calc_time_avg_rate(
        exp_cond,
        human_itv=True, clip_increase=True
) -> float:
    """
    change one key and fix others experimental conditions,
     then predict and plot
    :param clip_increase: if True, clip increase part and then mono decrease the start
    :param human_itv: whether human intervention
    :param exp_cond: dict, including Ta, va, w_rsm, vd
    :return: time_avg_rate
    """
    if clip_increase:
        assert human_itv
    model = gen_bag(ops='win')
    with open('../pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
        dct = pickle.load(pf)
    ft_scalar = dct['ft_scalar']
    x = ReaSdd.gen_sdd_data(**exp_cond)
    x_tilde = ft_scalar.transform(x)
    s = torch.zeros(x.shape[0], 2)
    s[[0, 15, 30, 45] + list(range(60, x.shape[0], 30)), 1] = 1
    ft = torch.from_numpy(x_tilde)
    ft_s = torch.cat((ft, s), dim=-1)
    ft, s, real_ind, length = mf.pack_test(ft_s)
    with torch.no_grad():
        ft = rnn.pad_sequence(ft, batch_first=True).cuda()
        pred, _ = model(ft, length)
    if human_itv:
        pred = mf.human_intervene(
            pred.cuda(), 1,
            torch.from_numpy(real_ind), s_min=1e-7,
            clip_increase=clip_increase
        ).squeeze(0).cpu().numpy()
    else:
        pred = pred.squeeze(0).cpu().numpy()
    if clip_increase:
        x = x[:pred.size, :]
    t = x[:, 0]
    pred = pred.flatten()
    rate = np.gradient(pred)
    time_avg_rate = simps(rate, t) / (t[-1] - t[0])
    return time_avg_rate


if __name__ == '__main__':
    ta_list = [ta + 273.15 for ta in [70, 80, 90, 100, 110]]
    va_list = [0.45, 0.6, 0.75, 0.9, 1.05]
    ws_list = [0.055, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2]
    vd_list = [1.0e-9, 1.25e-9, 1.5e-9, 1.75e-9, 2.0e-9]
    exp_conditions = tuple(product(ta_list, va_list, ws_list, vd_list))
    exp_names = ('Ta', 'va', 'w_rsm', 'vd')
    rates = []
    for cond in tqdm(exp_conditions):
        dct = dict(zip(exp_names, cond))
        dct['dur'] = 300
        rates.append(calc_time_avg_rate(dct))
    with open('../pickle_data/rate5x5x7x5.pkl', 'wb') as pickle_file1:
        pickle.dump((exp_conditions, rates), pickle_file1)
