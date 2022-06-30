# -*- coding: utf-8 -*-
# =============================================================================
# 付楠老师文章重复验证
# SR 和 Tp
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'  ##坐标轴是一个封闭的正方形##

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)  ##科学计数法##
formatter.set_powerlimits((-1, 1))
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
font_set1 = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=5)  ##让坐标轴显示中文##

# ------------------------------------------------------------------------------
Mair = 28.85
Mvapor = 18.0153
P = 101325.
RR = 8314.
pi = np.pi
Y = 0.0001
dp0 = 0.0015632
Up0 = 0.75  # 风速
Tp0 = 293.15
R = 8.314
NanFu = "NanFu"
Type3 = "Type3"
Type4 = "Type4"
Type5 = "Type5"
Final = "Final"
Regression = "Regression"
WaterCpCoeffs = [15341.1046350264, -116.019983347211, 0.451013044684985,
                 -0.000783569247849015, 5.20127671384957e-07]
WateRhoCoeffs = [98.343885, 0.30542, 647.13, 0.081]
HLCoeffs = [647.13, 2889425.47876769, 0.3199, -0.212, 0.25795, 0.]
AirMuCoeffs = [1.5061e-06, 6.16e-08, -1.819e-11, 0., 0.]
AirCpCoeffs = [948.76, 0.39171, -0.00095999, 1.393e-06, -6.2029e-10]
AirRhoCoeffs = [4.0097, -0.016954, 3.3057e-05, -3.0042e-08, 1.0286e-11]
AirKappaCoeffs = [0.0025219, 8.506e-05, -1.312e-08, 0., 0.]
VaporMuCoeffs = [1.5068e-06, 6.1598e-08, -1.8188e-11, 0, 0]
VaporRhoCoeffs = [2.5039, -0.010587, 2.0643e-05, -1.8761e-08, 6.4237e-12]
VaporKappaCoeffs = [0.0037972, 0.00015336, -1.1859e-08, 0, 0]
VaporCpCoeffs = [1563.1, 1.604, -0.0029334, 3.2168e-06, -1.1571e-09]

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 0.4149804174919806 [791697538462443.2, 70461.64534519543, -30.830795231152955, 498.15795740478313, -79.05596698883657]
# 0.4282333678168537 [443930816463263.25, 64213.80156264765, -41.042446037847895, 466.9524431758325, -5.444550345610587]
SRCoeff_NanFu = [1.214e+14, 1.17e+05, 0]
SRCoeff_NanFu1 = [12086895.43601252, 58263.04008944, 0, 0]
SRCoeff_NanFu2 = [50148012578006.13, 101297.87718138, 0, 0]
SRCoeff_Type3 = [3.205e+11, 88055, 0.3821]
SRCoeff_Type4 = [2.204e+12, 93260, -41.81]
SRCoeff_Type5 = [5.602e10, 83238, 0.23369, -11.1]
SRCoeff_Final = [7.324e+8, 69536, -1.605, 47.24, 1.625]  # 0.047675338933235265
# SRCoeff_Final  = [2.2309351084739704e-20, -4.898972353229335, -121950.15740769736, 0.08546565162705029, -0.029659117105919838]
SRCoeff_Regression = [8.951979238874504e-21, -4.012033360942845, -124608.97482805695]
# ------------------------------------------------------------------------------
VarList = lambda x: [1., x, x ** 2, x ** 3, x ** 4]


def calcWaterRho(T):
    '''
    计算温度T(K)下水的密度(kg/m3)
    '''
    a_, b_, c_, d_ = WateRhoCoeffs
    return a_ / pow(b_, 1 + pow(1 - T / c_, d_))


def calcMilkRho(T):
    '''
    计算温度T下牛奶的密度
    '''
    return 1435.


def calcWaterHL(T):
    '''
    计算温度T下的水的蒸发焓(J/kg)
    '''
    Tc_, a_, b_, c_, d_, e_ = HLCoeffs
    Tr = T / Tc_
    return a_ * pow(1 - Tr, ((e_ * Tr + d_) * Tr + c_) * Tr + b_)


def calcDropletCp(T, py):
    '''
    计算温度T下固含量为py的脱脂奶的比热容(J/(kg·K))
    '''
    # CpWater = np.dot(WaterCpCoeffs ,VarList(T))
    T = T - 273.15
    CpLactose = 1548.84 + 1.9625 * T - 0.0059399 * T ** 2.
    CpProtein = 2008.2 + 1.2089 * T - 0.0013129 * T ** 2.
    CpFat = 1984.2 + 1.4733 * T - 0.0048008 * T ** 2.
    CpMineral = 1092.6 + 1.8896 * T - 0.0036817 * T ** 2.
    cpMilk = CpLactose * 0.498 + CpProtein * 0.365 + CpFat * 0.006 + CpMineral * 0.093  # + CpWater*0.038
    # Cpd = py * CpWater + (1 - py) * cpMilk
    # print(Cpd)
    return cpMilk


def calcPhysicalProperities(Tf, Y):
    '''
    计算膜温度Tf(K)下水蒸气质量分数为Y的湿空气的密度、粘度(Pa·s)、比热容、热传导系数W/(m·K)
    '''
    TfList = [1., Tf, Tf ** 2, Tf ** 3, Tf ** 4]
    Mmixture = 1. / (Y / Mvapor + (1 - Y) / Mair)
    rho = P * Mmixture / (RR * Tf)  # 密度
    mu = (1 - Y) * np.dot(AirMuCoeffs, TfList) + Y * np.dot(VaporMuCoeffs, TfList)  # 黏度
    kappa = (1 - Y) * np.dot(AirKappaCoeffs, TfList) + Y * np.dot(VaporKappaCoeffs, TfList)  # 热传导系数
    cp = ((1 - Y) * np.dot(AirCpCoeffs, TfList) + Y * np.dot(VaporCpCoeffs, TfList))  # cp

    return [rho, mu, cp, kappa]


def calc_dSRdt(SR, px, dxdt, dTpdt, Tp, SRCoeff, Type):
    '''
    计算当前状态下存活率随时间的变化率dSRdt
    '''
    c_ = 0
    b_ = 0
    k0_, Ed_, a_, = SRCoeff[:3]
    if SRCoeff[4:]:
        b_, = SRCoeff[4:]
    if SRCoeff[5:]:
        c_, = SRCoeff[5:]
    RT = 8.314 * Tp
    dxdt = np.abs(dxdt)
    dTpdt = np.abs(dTpdt)
    if Type == "NanFu":
        kd = k0_ * np.exp(-Ed_ / RT)
        return -kd * SR
    elif Type == "Type4":
        kd = k0_ * np.exp(-Ed_ / RT) * (1 + b_ * dxdt)
        return -kd * SR
    elif Type == "Type5":
        kd = k0_ * np.exp(a_ * px - Ed_ / RT) * (1 + b_ * dxdt)
        return -kd * SR
    elif Type == "Type3":
        kd = k0_ * np.exp(a_ * px - Ed_ / RT)
        return -kd * SR
    elif Type == "Final":
        kd = k0_ * np.exp(a_ * px - Ed_ / RT) * (1 + b_ * dTpdt) * (1 + c_ * dxdt)
        return -kd * SR
    elif Type == "Regression":
        kd = k0_ * np.exp(a_ * px - Ed_ / RT)
        return -kd * SR
    else:
        return 0


def SMSDD(l, w, Tb0, Tp0, pMass0, dp0, Up0, ms, pS0, Y, SRCoeff, Type):
    '''
    计算单液滴脱脂奶干燥主程序
    '''
    # 活化能特征方程要用
    a = 0.998
    b = 1.405
    d = 0.930

    Mo = 0.06156
    Co = 0.001645  # GAB
    Hi = 24831.
    Ko = 5.71  # GAB
    Hii = -5118
    TpNow, pMass, SR = w
    Mw = [Mair, Mvapor]
    Ts = (2. * TpNow + Tb0) / 3.
    px = (pMass - ms) / ms  # 液滴、颗粒的干基含水率
    py = (pMass - ms) / pMass  # 液滴、颗粒中水分的质量分数
    diameter = 0
    if pS0 == 0.1:
        deltaMpByRhoWater = (pMass0 - pMass) / calcWaterRho(TpNow)
        diameter = ((np.pi / 6. * dp0 ** 3. - deltaMpByRhoWater) * 6. / np.pi) ** (1. / 3.)
    elif pS0 == 0.2:
        diameter = dp0 * (0.59 + 0.41 * (px / ((1 - pS0) / pS0)))
    elif pS0 == 0.3:
        diameter = dp0 * (0.69 + 0.31 * (px / ((1 - pS0) / pS0)))
    else:
        print("error \n")

    Ap = pi * diameter ** 2.

    fPsat = np.exp(23.365 - 3919.9 / (Tb0 - 42.062))  # 算RH用
    Psat = np.exp(23.365 - 3919.9 / (TpNow - 42.062))  # 算RH
    AH = Y / (1 - Y)  # 绝对湿度
    RH = (P * AH) / (0.622 * fPsat + AH * fPsat)  # 相对湿度
    # ----------------------------------------
    C = Co * np.exp(Hi / (8.314 * Tb0))
    K = Ko * np.exp(Hii / (8.314 * Tb0))
    pxe = (C * K * Mo * RH) / ((1 - K * RH) * (1 - K * RH + C * K * RH))
    lnRH = np.log(RH)
    deltaEve = -(Tb0 * 8314. * lnRH)  # 平衡活化能

    Yavg = Y
    nx = px - pxe
    if nx >= 0:
        deltaEv = 0
        if pS0 == 0.1:
            deltaEv = deltaEve * a * np.exp(-b * nx ** d)
        elif pS0 == 0.2:
            deltaEv = deltaEve * a * np.exp(-b * nx ** d)
        elif pS0 == 0.3:
            # px - pxe 应该可以写成 nx 啊
            deltaEv = deltaEve * (3.0318e-2 * (px - pxe) ** 4 - 0.26637 * (px - pxe) ** 3
                                  + 0.85762 * (px - pxe) ** 2 - 1.3635 * (px - pxe) + 0.99609)

        Pvs = Psat * np.exp(-deltaEv / (8314. * TpNow))  # 先算表面的压力，再在下行换算成密度
        rhoS = Pvs * Mvapor / 8314. / TpNow  # rho = pM/RT
        Yw = (Pvs / P * Mvapor) / (Pvs / P * Mair + (1 - Pvs / P) * Mair)
        Yavg = (Yw + Y) / 2.

    rho, mu, cp, kappa = calcPhysicalProperities(Ts, Yavg)
    Xvapor = Y / Mw[1] / (Y / Mw[1] + (1 - Y) / Mw[0])
    Dab = 3.564e-10 * ((Ts * 2.0) ** 1.75)
    Cpd = calcDropletCp(TpNow, py)
    HL = calcWaterHL(TpNow)

    Re = diameter * Up0 * rho / mu
    Sc = mu / rho / Dab
    Pr = mu * cp / kappa
    Nu = 2.04 + 0.62 * Re ** 0.5 * Pr ** (1. / 3.)
    Sh = 1.63 + 0.54 * Re ** 0.5 * Sc ** (1. / 3.)

    hm = Sh * Dab / diameter
    h = Nu * kappa / diameter
    dmdt = 0.0
    rhoF = Xvapor * P * Mvapor / (8314. * Tb0)  # rho_{v,b}
    if nx >= 0:
        dmdt = hm * Ap * (rhoS - rhoF)
    dxdt = dmdt / ms
    dTpdt = (h * (Tb0 - TpNow) * Ap - (dmdt * HL)) / (pMass * Cpd)

    dSRdt = calc_dSRdt(SR, px, dxdt, dTpdt, TpNow, SRCoeff, Type=Type)
    return [dTpdt, -dmdt, dSRdt]


def dataRead():
    '''
    读取文献取点益生菌数据
    '''
    data0 = pd.read_csv(r'70-20.csv', header=None)
    data1 = pd.read_csv(r'90-20.csv', header=None)
    data2 = pd.read_csv(r'110-20.csv', header=None)
    data3 = pd.read_csv(r'70-10.csv', header=None)
    data4 = pd.read_csv(r'90-10.csv', header=None)
    data5 = pd.read_csv(r'110-10.csv', header=None)
    for data in [data0, data1, data2, data3, data4, data5]:
        data.columns = ['Time (s)', 'SR']
        data['SR'] /= data['SR'][0]
        data['Time (s)'] = round(data['Time (s)'])
    return data0, data1, data2, data3, data4, data5


def TpRead():
    '''
    读取文献取点液滴温度数据
    '''
    data0 = pd.read_csv(r'70-20Tp.csv', header=None)
    data1 = pd.read_csv(r'90-20Tp.csv', header=None)
    data2 = pd.read_csv(r'110-20Tp.csv', header=None)
    data3 = pd.read_csv(r'70-10Tp.csv', header=None)
    data4 = pd.read_csv(r'90-10Tp.csv', header=None)
    data5 = pd.read_csv(r'110-10Tp.csv', header=None)
    for data in [data0, data1, data2, data3, data4, data5]:
        data.columns = ['Time (s)', 'TP']
    return data0, data1, data2, data3, data4, data5


def PxRead():
    '''
    读取文献取点液滴含水率数据
    '''
    data0 = pd.read_csv(r'70-20PX.csv', header=None)
    data1 = pd.read_csv(r'90-20PX.csv', header=None)
    data2 = pd.read_csv(r'110-20PX.csv', header=None)
    data3 = pd.read_csv(r'70-10PX.csv', header=None)
    data4 = pd.read_csv(r'90-10PX.csv', header=None)
    data5 = pd.read_csv(r'110-10PX.csv', header=None)
    for data in [data0, data1, data2, data3, data4, data5]:
        data.columns = ['Time (s)', 'PX']
    return data0, data1, data2, data3, data4, data5


def SRExperiment():
    '''
    文献原始益生菌数据
    '''
    data = [
        (0, 1.00E+00, 1.00E+00, 1.00E+00),
        (15, 7.75E-01, 9.24E-01, 1.16E+00),
        (30, 1.08E+00, 1.04E+00, 1.30E+00),
        (45, 7.86E-01, 8.74E-01, 1.30E+00),
        (60, 1.06E+00, 9.24E-01, 1.25E+00),
        (90, 1.04E+00, 8.07E-01, 1.09E-01),
        (120, 8.77E-01, 7.39E-01, 8.73E-05),
        (150, 6.61E-01, 1.52E-01, 1.73E-06),
        (180, 9.69E-01, 1.92E-02, 1.73E-06),
        (210, 6.72E-01, 1.92E-03, 1.73E-06),
        (240, 3.72E-01, 5.71E-04, 1.73E-06),
        (270, 3.29E-01, 4.03E-04, 1.73E-06),
        (300, 3.20E-01, 2.69E-04, 1.73E-06)
    ]
    return np.array(data)


def ObjFunc(SRCoeff, Type=None):
    '''
    计算每组模拟值与实验值的均方误差MSE的和
    '''
    # SR0, SR1, SR2, SR3, SR4, SR5 = dataRead()
    SRdata = SRExperiment()
    SR0 = SRdata[:, 1]
    SR1 = SRdata[:, 2]
    SR2 = SRdata[:, 3]
    Time = SRdata[:, 0]
    TP0, TP1, TP2, TP3, TP4, TP5 = TpRead()
    PX0, PX1, PX2, PX3, PX4, PX5 = PxRead()
    SRdata = SRExperiment()
    SR0 = SRdata[:, 1]
    SR1 = SRdata[:, 2]
    SR2 = SRdata[:, 3]
    pS0 = 0.2  # 固含量 wt%
    rhoDrop0 = (1 - pS0) * calcWaterRho(Tp0) + pS0 * calcMilkRho(Tp0)  # 液滴初始密度
    pMass0 = dp0 ** 3 * pi / 6. * rhoDrop0
    ms = pMass0 * pS0
    w0 = [Tp0, pMass0, 1.0]
    addUp = []
    for Tb0, SR_ in zip([70 + 273.15, 90 + 273.15, 110 + 273.15], [SR0, SR1, SR2]):
        time = [0.]
        result = [w0]
        r0 = ode(SMSDD).set_integrator('lsoda', method='bdf')
        r0.set_initial_value(w0, 0).set_f_params(Tb0, Tp0, pMass0, dp0, Up0, ms, pS0, Y, SRCoeff, Type)
        while r0.successful() and r0.t < 305.:
            r0.integrate(r0.t + 1)
            time.append(r0.t)
            result.append(r0.y)
        result = np.array(result)
        a, b = result.shape
        if a < 304 or np.max(result[:, 2]) > 1.1:
            return 1000
        else:
            calc_value = [result[int(d), 2] for d in Time]
            addUp.append(np.mean((calc_value - SR_) ** 2))
    return np.sum(addUp)


def draw_pic1(SRCoeff, Type):
    '''
    对计算值和实验值绘图
    '''
    # SR0, SR1, SR2, SR3, SR4, SR5 = dataRead()
    SRdata = SRExperiment()
    SR0 = SRdata[:, 1]
    SR1 = SRdata[:, 2]
    SR2 = SRdata[:, 3]
    time = SRdata[:, 0]
    TP0, TP1, TP2, TP3, TP4, TP5 = TpRead()  # 液滴温度
    PX0, PX1, PX2, PX3, PX4, PX5 = PxRead()  # 含水量
    pS0 = 0.2  # 固含量 wt%
    rhoDrop0 = (1 - pS0) * calcWaterRho(Tp0) + pS0 * calcMilkRho(Tp0)  # 初始颗粒密度
    pMass0 = dp0 ** 3 * pi / 6. * rhoDrop0  # 颗粒质量
    ms = pMass0 * pS0  # 固形物质量
    w0 = [Tp0, pMass0, 1.0]
    fig, axes = plt.subplots(2, 2, dpi=250)
    for SR_, TP_, PX_, Tb_ in zip([SR0, SR1, SR2], [TP0, TP1, TP2], [PX0, PX1, PX2], [70, 90, 110]):
        label = 'Tb = ' + str(Tb_) + '$^{\circ}$C'
        axes[0, 0].scatter(time, SR_, s=2, label=label)
        axes[0, 1].scatter(TP_['Time (s)'], TP_['TP'], s=2, label=label)
        axes[1, 0].scatter(PX_['Time (s)'], PX_['PX'], s=2, label=label)
    for Tb0 in [70 + 273.15, 90 + 273.15, 110 + 273.15]:
        time = [0.]
        result = [w0]
        r0 = ode(SMSDD).set_integrator('lsoda', method='bdf')
        #  Y是水蒸气质量分数, SRCoeff是kinetic model里的参数
        r0.set_initial_value(w0, 0).set_f_params(Tb0, Tp0, pMass0, dp0, Up0, ms, pS0, Y, SRCoeff, Type)
        while r0.successful() and r0.t < 300.:
            r0.integrate(r0.t + 1)
            time.append(r0.t)
            result.append(r0.y)
        result = np.array(result)
        a, b = result.shape
        if a < 294 or max(result[:, 2]) > 1.1:
            return np.nan
        else:
            label = 'Tb = ' + str(Tb0 - 273.15) + '$^{\circ}$C'
            axes[0, 0].plot(time, result[:, 2], label=label)
            axes[0, 1].plot(time, result[:, 0] - 273.15, label=label)
            axes[1, 0].plot(time, result[:, 1] / ms - 1, label=label)
    axes[0, 0].set_xlabel("干燥时间（s）", fontproperties=font_set)
    axes[0, 0].set_ylabel("益生菌存活率", fontproperties=font_set)
    axes[0, 0].set_xlim(0, 310)
    axes[0, 0].set_ylim(0, 1.5)
    # axes[0,0].legend(loc=0,prop=font_set1)

    axes[0, 1].set_xlabel("干燥时间（s）", fontproperties=font_set)
    axes[0, 1].set_ylabel("液滴的温度（℃）", fontproperties=font_set)
    axes[0, 1].set_xlim(0, 310)
    axes[0, 1].set_ylim(20, 120)
    axes[1, 0].set_xlabel("干燥时间（s）", fontproperties=font_set)
    axes[1, 0].set_ylabel("液滴的干基含水量", fontproperties=font_set)
    axes[1, 0].set_xlim(0, 310)
    axes[1, 0].set_ylim(0, 4)
    # axes[0,1].legend(loc=0,prop=font_set1)
    # axes[1,0].legend(loc=0,prop=font_set1)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)

    # =============================================================================
    #     pS0 = 0.1
    #     rhoDrop0 = (1 - pS0)*calcWaterRho(Tp0) + pS0*calcMilkRho(Tp0)
    #     pMass0 = dp0 ** 3 * pi / 6. * rhoDrop0
    #     ms = pMass0 * pS0
    #     w0 = [Tp0,pMass0,1.0]
    #
    #     fig1,axes1 = plt.subplots(2,2,dpi=250)
    #     for SR_,TP_,PX_ in zip([SR3,SR4,SR5],[TP3,TP4,TP5],[PX3,PX4,PX5]):
    #         axes1[0,0].scatter(SR_['Time (s)'],SR_['SR'],s=2)
    #         axes1[0,1].scatter(TP_['Time (s)'],TP_['TP'],s=2)
    #         axes1[1,0].scatter(PX_['Time (s)'],PX_['PX'],s=2)
    #
    #     for Tb0 in [70+273.15,90+273.15,110+273.15]:
    #         time = [0.]
    #         result = [w0]
    #         r0 = ode(SMSDD).set_integrator('lsoda', method='bdf')
    #         r0.set_initial_value(w0, 0).set_f_params(Tb0,Tp0,pMass0,dp0,Up0,ms,pS0,Y,SRCoeff,Type)
    #         while r0.successful() and r0.t < 300.:
    #             r0.integrate(r0.t+1)
    #             time.append(r0.t)
    #             result.append(r0.y)
    #         result = np.array(result)
    #         a,b = result.shape
    #         if a < 294 or max(result[:,2])>1.1 :
    #             return np.nan
    #         else:
    #             label='Tb = '+str(Tb0-273.15)+'$^{\circ}$C'
    #             axes1[0,0].plot(time,result[:,2],label= label)
    #             axes1[0,1].plot(time,result[:,0]-273.15,label= label)
    #             axes1[1,0].plot(time,result[:,1]/ms-1,label= label)
    # =============================================================================
    sns.despine()
    plt.show()


# main()
if __name__ == "__main__":
    # =============================================================================
    #     from sklearn import linear_model
    #     T = np.arange(273.15,373.15,1)
    #     Cp = calcDropletCp(T,0)
    #     #Y1 = T*-2.57980445e-01 +T**2*2.98202339e-04 +T**3*-6.76183957e-07 + 1063.323888055272
    #
    #     #2264698
    #
    #     reg = linear_model.LinearRegression()
    #     X = np.array([T,T**2]).T
    #     reg.fit(X,Y)
    #     print(reg.coef_)
    #     print(reg.intercept_)
    #     print(reg.score(X,Y))
    #     plt.plot(T,Y)
    # =============================================================================

    # =============================================================================
    #     pS0 = 0.2
    #     Tp0 = 298.15
    #     dp0 = 1.6e-4
    #     rhoDrop0 = (1 - pS0)*calcWaterRho(Tp0) + pS0*calcMilkRho(Tp0)
    #     pMass0 = dp0 ** 3 * pi / 6. * rhoDrop0
    #     ms = pMass0 * pS0
    #     print(ms)
    # =============================================================================
    draw_pic1(SRCoeff_NanFu, Type=NanFu)
# =============================================================================
#     draw_pic1(SRCoeff_NanFu1,Type=NanFu)
#     draw_pic1(SRCoeff_Type5,Type=Type5)
#     draw_pic1(SRCoeff_Type4,Type=Type4)
#     draw_pic1(SRCoeff_Type3,Type=Type3)
#     draw_pic1(SRCoeff_Final,Type=Final)
# =============================================================================
# draw_pic1(SRCoeff_Regression,Type=Regression)
# draw_pic1(SRCoeff_NanFu,Type=NanFu)
# draw_pic1(SRCoeff_Final,Type=Final)
# print(ObjFunc(SRCoeff_Final,Type=Final))
# print(ObjFunc(SRCoeff_NanFu,Type=NanFu))
# print(ObjFunc(SRCoeff_NanFu1,Type=NanFu))
# print(ObjFunc(SRCoeff_Final,Type=Final))
