from matplotlib import pyplot as plt
from pred_fun import vary_pred_plot
import pickle

ta_base = 90
va_base = 0.7
ws_base = 0.1
vd_base = 1.5e-9
with open('../pickle_data/sns-ana-time-0d1.pkl', 'wb') as pkl_time_avg:
    time_avg_rate = {}
    # # extend time & vary Ta
    # ta_list = [70, 75, 80, 85, 90, 95, 100, 105, 110]
    # fig_extend, ax, _, _, _ = vary_pred_plot('Ta', {
    #     'Ta': list(map(lambda x: x + 273.15, ta_list)),
    #     'va': va_base,
    #     'w_rsm': ws_base,
    #     'vd': vd_base,
    #     'dur': [350]
    # }, labels=list(map(lambda x: r'$' + str(
    #     x) + r'\mathrm{^\circ\hspace{-0.25}C}$', ta_list)),
    #     clip_increase=False, human_itv=False)
    # fig_extend.savefig('../figures/vary_inp/extend-time.png')

    #  vary Ta
    ta_list = [70, 75, 80, 85, 90, 95, 100, 105, 110]
    fig_Ta, ax, time_avg_rate['Ta'], fig_Ta_rate, ax_Ta_rate = \
        vary_pred_plot('Ta', {
            'Ta': list(map(lambda x: x + 273.15, ta_list)),
            'va': va_base,
            'w_rsm': ws_base,
            'vd': vd_base,
            'dur': [300]
        }, labels=list(map(lambda x: r'$' + str(
            x) + r'\mathrm{^\circ\hspace{-0.25}C}$', ta_list
                           )), clip_increase=True)
    fig_Ta.savefig('../figures/vary_inp/Ta.png', transparent=True)
    fig_Ta_rate.savefig('../figures/vary_inp/Ta-rate.png', transparent=True)
    # #  enlarge Ta
    # ax[1, 0].set_xlim([-1, 131])
    # ax[1, 0].set_ylim([0.84, 1.04])
    # plt.subplots_adjust(left=0.3,wspace=0.8)
    # fig_Ta.savefig('../figures/vary_inp/Ta-enlarge.png')
    # #  Label Ta
    # plt.rc('font', family='Times New Roman', size=19)
    # colors = plt.cm.tab10.colors
    # fig_exp, axes = plt.subplots(2, 2)
    # ax = axes[1, 0]
    # font_formula = dict(math_fontfamily='cm', size=26)
    # font_text = dict(family='Times New Roman', size=23)
    # font_legend = {'size': 24, 'math_fontfamily': 'cm'}
    # s1, s2, s3 = read_temp_x([1, 2, 3])
    # ax.plot(s1[:, 0], s1[:, 1], label=r'$70^\circ\hspace{-0.25}\mathrm{C}$',
    #         color=colors[0], markevery=marker_sep, marker=markers[0])
    # ax.plot(s2[:, 0], s2[:, 1], label=r'$90^\circ\hspace{-0.25}\mathrm{C}$', color=colors[2])
    # ax.plot(s3[:, 0], s3[:, 1], label=r'$110^\circ\hspace{-0.25}\mathrm{C}$', color=colors[4])
    # ax.set_xlabel('Time (s)', fontdict=font_text)
    # ax.set_ylabel(r'$s$', fontdict=font_formula)
    # ax.legend(loc=(0.46, 0.46), prop=font_legend)
    # ax.set_xlim([-1, 131])
    # ax.set_ylim([0.81, 1.04])
    # fig_exp.set_size_inches([15.36, 7.57])
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.8)
    # fig_exp.savefig('../figures/vary_inp/Ta-label.png')

    # vary va
    va_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    fig_va, ax_va, time_avg_rate['va'], fig_va_rate, ax_va_rate = \
        vary_pred_plot('va', {
            'Ta': ta_base + 273.15,
            'va': va_list,
            'w_rsm': ws_base,
            'vd': vd_base,
            'dur': [300]
        }, labels=list(map(lambda x: r'$' + str(
            x) + r'~\mathrm{m/s}$', va_list)), human_itv=True, clip_increase=True)
    plt.subplots_adjust(left=0.174)
    fig_va.savefig('../figures/vary_inp/va.png')
    fig_va_rate.savefig('../figures/vary_inp/va-rate.png', transparent=True)

    # # vary ws
    ws_list = [0.050, 0.055, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.25]
    fig_w_rsm, ax_ws, time_avg_rate['ws'], fig_w_rsm_rate, ax_w_rsm_rate = \
        vary_pred_plot('w_rsm', {
            'Ta': ta_base + 273.15,
            'va': va_base,
            'w_rsm': ws_list,
            'vd': vd_base,
            'dur': [300]
        }, labels=['5 wt%', '5.5 wt%', '6 wt%', '7 wt%', '8 wt%',
                   '10 wt%', '15 wt%', '20 wt%', '25 wt%'
                   ], human_itv=True, clip_increase=True)
    plt.subplots_adjust(left=0.165)
    fig_w_rsm.savefig('../figures/vary_inp/w-rsm.png', transparent=True)
    fig_w_rsm_rate.savefig('../figures/vary_inp/ws-rate.png', transparent=True)
    # enlarge w_rsm
    # ax[1, 1].set_xlim([231, 289])
    # ax[1, 1].set_ylim([-3.8, -3.35])
    # ax[0, 0].set_xlim([157.32, 182.35])
    # ax[0, 0].set_ylim([81.22, 84.87])
    # ax[0, 1].set_xlim([157.32, 182.35])
    # ax[0, 1].set_ylim([0.32, 0.46])
    # plt.subplots_adjust(wspace=0.8, right=0.64)
    # fig_w_rsm.savefig('../figures/vary_inp/wrsm-enlarge.png')

    # # vary vd
    vd_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    fig_vd, ax_vd, time_avg_rate['vd'], fig_vd_rate, ax_vd_rate = \
        vary_pred_plot('vd', {
            'Ta': ta_base + 273.15,
            'va': va_base,
            'w_rsm': ws_base,
            'vd': list(map(lambda x: x * 1e-9, vd_list)),
            'dur': [300]
        }, labels=list(map(lambda x: r'$' + str(
            x) + r'~\mathrm{\mu L}$', vd_list)), human_itv=True, clip_increase=True)
    fig_vd.savefig('../figures/vary_inp/vd.png')
    fig_vd_rate.savefig('../figures/vary_inp/vd-rate.png', transparent=True)
    #
    pickle.dump((time_avg_rate, {
        'Ta': ta_list, 'va': va_list, 'ws': ws_list, 'vd': vd_list
    }), pkl_time_avg)
