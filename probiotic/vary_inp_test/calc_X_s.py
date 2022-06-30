from probioticsCCE.probiotic.gen_bag_model import gen_bag
import numpy as np
import probioticsCCE.my_functions as mf
import pickle
import torch
from torch.nn.utils import rnn
from probioticsCCE.probiotic. \
    drying_kinetics import zhuh_REA_SDD as ReaSdd
from scipy.integrate import simps
from itertools import product
from tqdm import tqdm


# calc X and s at the time instant t


def calc_x_s(
        exp_cond,
        ops='win'
) -> (float, float):
    """
    change one key and fix others experimental conditions,
     then predict and plot
    :param ops: operating system
    :param exp_cond: dict, including Ta, va, w_rsm, vd
    :return: x_t and s_t
    """
    model = gen_bag(ops=ops)
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
    pred = mf.human_intervene(
        pred.cuda(), 1,
        torch.from_numpy(real_ind), s_min=1e-7,
        clip_increase=True
    ).squeeze().cpu().numpy()
    if pred.size - 1 < exp_cond['dur']:
        return 1000, -1
    else:
        x = x[:pred.size, 2]
    return x[-1], pred[-1]


if __name__ == '__main__':
    x, s = calc_x_s({
        'Ta': 110 + 273.15, 'va': 0.7, 'w_rsm': 0.2, 'vd': 1.5e-9, 'dur': 270
    })
