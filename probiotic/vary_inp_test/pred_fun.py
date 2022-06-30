from probioticsCCE.probiotic.gen_bag_model import gen_bag
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm
import probioticsCCE.my_functions as mf
import pickle
import torch
from torch.nn.utils import rnn
from copy import deepcopy as dc
from probioticsCCE.probiotic. \
    drying_kinetics import zhuh_REA_SDD as ReaSdd
from scipy.integrate import simps

markers = ['o', 'v', '*', 'd', 's', '>', '|', '^', 'x', '.']
markersizes = [5, 5, 7, 5, 5, 5, 9, 5, 6, 10]
marker_sep = 20


def vary_pred_plot(
        vary_param, exp_cond, labels, ops='win',
        human_itv=True, clip_increase=True
):
    """
    change one key and fix others experimental conditions,
     then predict and plot
    :param clip_increase: if True, clip increase part and then mono decrease the start
    :param human_itv: whether human intervention
    :param labels: used in legend
    :param vary_param: the param (key) to be changed
    :param exp_cond: dict, including Ta, va, w_rsm, vd
    :param ops: 'win' or 'linux', useful when gen_bag
    :return: pyplot.fig
    """
    time_avg_rate = []
    if clip_increase:
        assert human_itv
    model = gen_bag(ops='win')
    font_text = {'family': 'Times New Roman', 'size': 21}
    font_title = {'family': 'Times New Roman', 'size': 23}
    font_formula = fm.FontProperties(
        math_fontfamily='cm', size=21)
    if '$' in labels[0]:
        font_legend = {'size': 19, 'math_fontfamily': 'cm'}
    else:
        font_legend = {'size': 19}
    with open('../pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
        dct = pickle.load(pf)
    ft_scalar = dct['ft_scalar']
    plt.rc('font', family='Times New Roman', size=17)
    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(
        hspace=0.526,
        left=0.18, right=0.971,
        top=0.974, bottom=0.138
    )
    fig1, ax1 = plt.subplots()  # calc inactivation rate
    ax1.set_xlabel('Time (s)', fontdict=font_text)
    ax1.set_ylabel(r'$\Delta s/\Delta t$', fontproperties=font_formula)
    y_title = -0.4
    ax[1, 0].set_xlabel('Time (s)', fontdict=font_text)
    ax[1, 0].set_ylabel(r'$s$', fontproperties=font_formula)
    # ax[1, 0].set_title('(c) linear survival rate', fontdict=font_title, y=y_title)
    ax[1, 1].set_xlabel('Time (s)', fontdict=font_text)
    ax[1, 1].set_ylabel(r'$\lg s$', fontproperties=font_formula)
    # ax[1, 1].set_title('(d) log survival rate', fontdict=font_title, y=y_title)
    ax[0, 0].set_xlabel('Time (s)', fontdict=font_text)
    ax[0, 0].set_ylabel(r'$T_\mathrm{d}~ (^\circ\hspace{-0.25}\mathrm{C})$',
                        fontproperties=font_formula)
    # ax[0, 0].set_title('(a) droplet temperature', fontdict=font_title, y=y_title)
    ax[0, 1].set_xlabel('Time (s)', fontdict=font_text)
    ax[0, 1].set_ylabel(r'$X~(\mathrm{kg/kg})$', fontproperties=font_formula)
    # fig0, ax0 = plt.subplots()  # use in paper structure
    # ax[0, 1].set_title('(b) moisture content on the dry basis', fontdict=font_title, y=y_title)
    for i, vary_val in enumerate(exp_cond[vary_param]):
        label = labels[i]
        marker = markers[i]
        markersize = markersizes[i]
        tmp = dc(exp_cond)
        if len(exp_cond['dur']) == 1:
            tmp['dur'] = exp_cond['dur'][0]
        else:
            tmp['dur'] = exp_cond['dur'][i]
        tmp[vary_param] = vary_val
        if vary_val - 273.15:
            x = ReaSdd.gen_sdd_data(**tmp)
        else:  # vary_val==0, to compare with truth
            x = ReaSdd.read_temp_x([5])[0]
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
        ax[0, 0].plot(t, x[:, 1] - 273.15, label=label, marker=marker,
                      markersize=markersize, markevery=marker_sep)
        ax[0, 1].plot(t, x[:, 2], label=label, marker=marker,
                      markersize=markersize, markevery=marker_sep)
        pred = pred.flatten()
        rate = np.gradient(pred)
        time_avg_rate.append(simps(rate, t) / (t[-1] - t[0]))
        ax[1, 0].plot(t, pred, label=label, marker=marker,
                      markersize=markersize, markevery=marker_sep)
        # # ax[1, 0].plot(t[:-1], x[1:, 1] - x[:-1, 1], label=label) dT/dt
        ax[1, 1].plot(t, np.log10(pred), label=label, marker=marker,
                      markersize=markersize, markevery=marker_sep)
        ax1.plot(t, rate, marker=marker,
                 markersize=markersize, markevery=marker_sep)
        # ax0.plot(t, np.log10(pred.flatten()), label=label, lw=2, marker=marker,
        #          markersize=markersize * 1.5, markevery=marker_sep)  # use in paper structure
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', prop=font_legend)
    ax1.legend(handles, labels, loc='best', prop=font_legend)
    # ax[0, 0].legend(loc='best', prop=font_legend)
    # ax[0, 1].legend(loc='best', prop=font_legend)
    # ax[1, 0].legend(loc='best', prop=font_legend)
    # ax[1, 1].legend(loc='best', prop=font_legend)
    fig.set_size_inches([15.36, 7.57])
    fig1.set_size_inches([9.04, 6.64])
    return fig, ax, time_avg_rate, fig1, ax1


if __name__ == '__main__':
    fig = vary_pred_plot('w_rsm', {
        'Ta': 100 + 273.15,
        'va': 0.8,
        'w_rsm': [0.119,0.13,0.15],
        'vd': 1.4e-9,
        'dur': [100]
    }, labels=['10%', '15%', '20%', '25%', '30%'])
