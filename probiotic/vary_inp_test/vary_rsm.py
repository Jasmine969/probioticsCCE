import numpy as np
from matplotlib import pyplot as plt
from pred_fun import vary_pred_plot
import pickle

ta_base = 90
va_base = 0.75
ws_base = 0.1
vd_base = 2e-9
with open('../pickle_data/vary_cond_t02.pkl', 'wb') as pkl_t02:
    t02_dct = {}
    #  vary Ta
    ta_list = [60, 70, 80, 90, 100, 110]
    fig_Ta, ax_ta, t02_Ta = \
        vary_pred_plot('Ta', {
            'Ta': list(map(lambda x: x + 273.15, ta_list)),
            'va': va_base,
            'ws': ws_base,
            'vd': vd_base,
            'dur': [300]
        }, labels=list(map(lambda x: r'$' + str(
            x) + r'\mathrm{^\circ\hspace{-0.25}C}$', ta_list
                           )), clip_increase=False)
    fig_Ta.savefig('../figures/vary_inp/Ta.png', transparent=True)
    # fig_Ta_rate.savefig('../figures/vary_inp/Ta-rate.png', transparent=True)

    # vary va
    va_list = [0.45, 0.6, 0.75, 0.9, 1.05]
    fig_va, ax_va, t02_va = \
        vary_pred_plot('va', {
            'Ta': ta_base + 273.15,
            'va': va_list,
            'ws': ws_base,
            'vd': vd_base,
            'dur': [300]
        }, labels=list(map(lambda x: r'$' + str(
            x) + r'~\mathrm{m/s}$', va_list)), human_itv=True, clip_increase=False)
    # plt.subplots_adjust(left=0.174)
    # fig_va.savefig('../figures/vary_inp/va.png')
    # fig_va_rate.savefig('../figures/vary_inp/va-rate.png', transparent=True)

    # vary ws
    ws_list = [0.055, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2]
    fig_ws, ax_ws, t02_ws = \
        vary_pred_plot('ws', {
            'Ta': ta_base + 273.15,
            'va': va_base,
            'ws': ws_list,
            'vd': vd_base,
            'dur': [300]
        }, labels=['5 wt%', '5.5 wt%', '6 wt%', '7 wt%', '8 wt%',
                   '10 wt%', '15 wt%', '20 wt%', '25 wt%'
                   ], human_itv=True, clip_increase=False)
    # plt.subplots_adjust(left=0.165)
    # fig_ws.savefig('../figures/vary_inp/ws.png', transparent=True)
    # fig_ws_rate.savefig('../figures/vary_inp/ws-rate.png', transparent=True)
    # enlarge _s
    # plt.subplots_adjust(wspace=0.8, right=0.64)
    # ax_ws[1, 1].set_xlim([231, 289])
    # ax_ws[1, 1].set_ylim([-3.8, -3.35])
    # ax_ws[0, 0].set_xlim([155, 180])
    # ax_ws[0, 0].set_ylim([81.22, 84.87])
    # ax_ws[0, 1].set_xlim([155, 180])
    # ax_ws[0, 1].set_ylim([0.33, 0.46])
    # fig_ws.savefig('../figures/vary_inp/wrsm-enlarge.png')

    # vary vd
    vd_list = [0.5, 1, 1.5, 2, 2.5, 3.0]
    fig_vd, ax_vd, t02_vd = \
        vary_pred_plot('vd', {
            'Ta': ta_base + 273.15,
            'va': va_base,
            'ws': ws_base,
            'vd': list(map(lambda x: x * 1e-9, vd_list)),
            'dur': [300]
        }, labels=list(map(lambda x: r'$' + str(
            x) + r'~\mathrm{\mu L}$', vd_list)), human_itv=True, clip_increase=False)
    # fig_vd.savefig('../figures/vary_inp/vd.png')
    # fig_vd_rate.savefig('../figures/vary_inp/vd-rate.png', transparent=True)

    # standardization of the drying conditions
    def normalize(lst):
        ar = np.array(lst, dtype=np.float_)
        ar = (ar - np.min(ar)) / np.ptp(ar)
        return ar

    pickle.dump({
        'Ta': (normalize(ta_list), t02_Ta), 'va': (normalize(va_list), t02_va),
        'ws': (normalize(ws_list), t02_ws), 'vd': (normalize(vd_list), t02_vd)
    }, pkl_t02)

plt.show()
