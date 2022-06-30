import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.integrate import ode
import xlsxwriter

import pandas as pd
from sklearn.metrics import r2_score
from sklearn import linear_model

from NanFuVF import calc_dSRdt, calcPhysicalProperities
from NanFuVF import VarList, calcWaterRho, calcMilkRho, calcWaterHL, calcDropletCp
from NanFuVF import AirMuCoeffs, AirCpCoeffs, VaporMuCoeffs, VaporCpCoeffs
from NanFuVF import NanFu, Type3, Type4, Type5, Final, Regression
from NanFuVF import SRCoeff_NanFu, SRCoeff_NanFu1, SRCoeff_NanFu2, SRCoeff_Type3 \
    , SRCoeff_Type4, SRCoeff_Type5, SRCoeff_Final, SRCoeff_Regression

Mair = 28.85
Mvapor = 18.0153
P = 101325.
RR = 8314.
pi = np.pi
g = 9.8
Y = 0.01
dp0 = 1.55e-4
Dc = 0.36
vfr = 255.5 / 60000
mfr_ = vfr * 1.23
pmfr10 = 2.96e-05
pmfr20 = 2.96e-05
pmfr30 = 2.96e-05
Tp0_10 = 25 + 273.15
Tp0_20 = 25 + 273.15
Tp0_30 = 25 + 273.15


def calcCdRe(Re):
    """
    计算液滴运动阻力系数
    """
    return 24. * (1 + 1. / 6. * Re ** (2. / 3.))


def UseRHCalcVaporMassFrac(RH, Tinf):
    """
    利用相对湿度计算Tinf温度下水汽的质量分数
    """
    fPsat = np.exp(23.365 - 3919.9 / (Tinf - 42.062))
    AH = 0.622 * RH * fPsat / (P - RH * fPsat)
    Y = AH / (1 + AH)
    return Y


def UseVaporMassFracCalcRH(w):
    """
    利用水汽的质量分数计算Tinf温度下的相对湿度
    """
    Y, Tinf = w
    fPsat = np.exp(23.365 - 3919.9 / (Tinf - 42.062))
    AH = Y / (1 - Y)
    RH = (P * AH) / (0.6244 * fPsat + AH * fPsat)
    return RH


def SMSD(l, w, Tb0, Tp0, pMass0, dp0, Up0, ms, py0, Tinf, Y, hw, Dc, mfr, pmfr, SRCoeff, Type):
    """
    脱脂奶喷雾干燥计算主程序
    """
    #  kinetic model参数
    a = 0.998
    b = 1.405
    d = 0.930

    Mo = 0.06156
    Co = 0.001645  # GAB
    Hi = 24831.
    Ko = 5.71  # GAB
    Hii = -5118.
    TbNow, TpNow, pMass, UpNow, SR, YNow = w
    Mw = [Mair, Mvapor]
    Tf = (2. * TpNow + TbNow) / 3.
    TbList = VarList(TbNow)
    px = (pMass - ms) / ms  # 液滴、颗粒的干基含水率
    py = (pMass - ms) / pMass  # 液滴、颗粒中水分的质量分数
    diameter = 0
    # 以液滴初始固含量选择干燥动力学参数
    if py0 <= 0.1:
        deltaMpByRhoWater = (pMass0 - pMass) / calcWaterRho(TpNow)
        diameter = ((np.pi / 6. * dp0 ** 3. - deltaMpByRhoWater) * 6. / np.pi) ** (1. / 3.)
    elif py0 <= 0.2:
        diameter = dp0 * (0.59 + 0.41 * (px / ((1 - py0) / py0)))
    elif py0 <= 0.3:
        diameter = dp0 * (0.69 + 0.31 * (px / ((1 - py0) / py0)))
    else:
        print("error \n")
    Ap = pi * diameter ** 2.
    fPsat = np.exp(23.365 - 3919.9 / (TbNow - 42.062))
    Psat = np.exp(23.365 - 3919.9 / (TpNow - 42.062))

    AH = YNow / (1 - YNow)
    RH = (P * AH) / (0.6244 * fPsat + AH * fPsat)
    #  Y是水汽的质量分数，Xvapor是水汽的体积分数（摩尔分数）
    Xvapor = YNow / Mw[1] / (YNow / Mw[1] + (1 - YNow) / Mw[0])
    Dab = 3.564e-10 * ((Tf * 2.0) ** 1.75)  # 扩散系数
    Cpd = calcDropletCp(TpNow, py)  # 脱脂奶比热容
    HL = calcWaterHL(TpNow)  # 水的蒸发焓
    # -------------------------------------------------------------------------
    C = Co * np.exp(Hi / (8.314 * TbNow))  # GAB
    K = Ko * np.exp(Hii / (8.314 * TbNow))  # GAB
    pxe = (C * K * Mo * RH) / ((1 - K * RH) * (1 - K * RH + C * K * RH))  # GAB
    lnRH = np.log(RH)
    deltaEve = -(TbNow * 8314. * lnRH)  # 平衡活化能
    dmdt = 0.0
    Yavg = YNow
    nx = px - pxe
    rhoS = None
    # 算活化能
    if nx >= 0:
        deltaEv = 0
        if py0 <= 0.1:
            deltaEv = deltaEve * a * np.exp(-b * nx ** d)
        elif py0 <= 0.2:
            deltaEv = deltaEve * (
                    -6.47438e-3 * nx ** 5 + 8.86858e-2 * nx ** 4 - 0.471097 * nx ** 3
                    + 1.22317 * nx ** 2 - 1.62539 * nx + 1.0092)
        elif py0 <= 0.3:
            deltaEv = deltaEve * (3.0318e-2 * nx ** 4 - 0.26637 * nx ** 3
                                  + 0.85762 * nx ** 2 - 1.3635 * nx + 0.99609)

        Pvs = Psat * np.exp(-deltaEv / (8314. * TpNow))  # 先算液滴表面压力，下行再转换成密度
        rhoS = Pvs * Mvapor / 8314. / TpNow
        Yw = (Pvs / P * Mvapor) / (Pvs / P * Mair + (1 - Pvs / P) * Mair)
        Yavg = (2. * Yw + YNow) / 3.

    rho, mu, cp, kappa = calcPhysicalProperities(Tf, Yavg)
    # -------------------------------------------------------------------------
    Re = diameter * UpNow * rho / mu
    Sc = mu / rho / Dab
    Pr = mu * cp / kappa
    Nu = 2.04 + 0.62 * Re ** 0.5 * Pr ** (1. / 3.)
    Sh = 1.63 + 0.54 * Re ** 0.5 * Sc ** (1. / 3.)
    hm = Sh * Dab / diameter
    h = Nu * kappa / diameter
    rhoF = Xvapor * P * Mvapor / (8314. * TbNow)
    if nx >= 0:
        dmdt = hm * Ap * (rhoS - rhoF)
    if dmdt < 0:
        dmdt = 0.0
    dTpdt = (h * (TbNow - TpNow) * Ap - (dmdt * HL)) / (pMass * Cpd)
    dmdl = dmdt / UpNow

    dTpdl = dTpdt / UpNow

    Mxc = 1. / (YNow / Mvapor + (1 - YNow) / Mair)
    rhoc = P * Mxc / (RR * TpNow)
    muc = (1 - YNow) * np.dot(AirMuCoeffs, TbList) + YNow * np.dot(VaporMuCoeffs, TbList)
    Rec = diameter * UpNow * rhoc / muc
    CdRe = calcCdRe(Rec)
    rhoP = pMass / (pi / 6. * diameter ** 3.)
    tau = rhoP * diameter ** 2 * 4. / 3. / muc / CdRe
    Ub = mfr / rhoc / (pi / 4. * Dc)
    dUpdt = (Ub - UpNow) / tau + g
    dUpdl = dUpdt / UpNow
    cpc = ((1 - YNow) * np.dot(AirCpCoeffs, TbList) +
           YNow * np.dot(VaporCpCoeffs, TbList))
    theta = pmfr / pMass0
    dTbdl = (hw * (pi * Dc) * (TbNow - Tinf) + theta *
             (dmdl * HL + h * (TbNow - TpNow) * Ap / UpNow)) / (mfr * cpc)
    dxdt = dmdt / ms
    dSRdt = calc_dSRdt(SR, px, dxdt, dTpdt, TpNow, SRCoeff, Type=Type)
    dSRdl = dSRdt / UpNow
    dYdl = theta / mfr * dmdl
    if dSRdl > 0:
        dSRdl = 0.0
    return [-dTbdl, dTpdl, -dmdl, dUpdl, dSRdl, dYdl]


def value(Tb_, coeff_, Up_, hw=0, py0=0, Tinf=300, Y=0.01, Tp0=286.15, pmfr=0, SRCoeff=None, Type=None):
    """
    计算益生菌最终存活率
    """
    rhoDrop0 = (1 - py0) * calcWaterRho(Tp0) + py0 * calcMilkRho(Tp0)
    pMass0 = dp0 ** 3 * pi / 6. * rhoDrop0
    ms = pMass0 * py0

    Tb0, mfr0, Up0 = Tb_, coeff_ * mfr_, Up_
    w0 = [Tb0, Tp0, pMass0, Up0, 1.0, Y]
    r = [w0]
    l = [0.0]
    result = ode(SMSD).set_integrator('lsoda', method='bdf')
    result.set_initial_value(w0, 0).set_f_params(
        Tb0, Tp0, pMass0, dp0, Up0, ms, py0, Tinf, Y, hw, Dc, mfr0, pmfr, SRCoeff, Type)
    while result.successful() and result.t <= 3.:
        result.integrate(result.t + 0.01)
        l.append(result.t)
        r.append(result.y)
    r = np.array(r)
    srList = r[:, 4]
    result = srList[-1]
    return result


def valueList(Tb_, coeff_, Up_, hw=0, py0=0, Tinf=300, Y=0.01, Tp0=286.15, pmfr=0, SRCoeff=None, Type=None):
    """
    计算液滴的干燥历程
    """

    rhoDrop0 = (1 - py0) * calcWaterRho(Tp0) + py0 * calcMilkRho(Tp0)
    pMass0 = dp0 ** 3 * pi / 6. * rhoDrop0
    ms = pMass0 * py0

    Tb0, mfr0, Up0 = Tb_, coeff_ * mfr_, Up_
    w0 = [Tb0, Tp0, pMass0, Up0, 1.0, Y]
    r = [w0]
    l = [0.0]
    result = ode(SMSD).set_integrator('dopri5')
    result.set_initial_value(w0, 0).set_f_params(
        Tb0, Tp0, pMass0, dp0, Up0, ms, py0, Tinf, Y, hw, Dc, mfr0, pmfr, SRCoeff, Type)

    while result.successful() and result.t <= 3.:
        result.integrate(result.t + 0.01)
        l.append(result.t)
        r.append(result.y)
    r = np.array(r)
    tbList = r[:, 0]
    tpList = r[:, 1]
    pwList = (r[:, 2] - ms) / r[:, 2]
    pxList = (r[:, 2] - ms) / ms

    srList = r[:, 4]
    YList = r[:, 5]
    UpList = r[:, 3]
    deltaTime = 0.01 / UpList
    Time = np.cumsum(deltaTime)
    return [pxList, srList, pwList, tbList, tpList, YList, l, Time]


def diffWithExp(SRCoeff, Type=None):
    """
    益生菌最终存活率的计算值与实验值的均方误差的和
    """
    # =============================================================================
    #     calc_value = value(
    #         375.15, 1, 1,hw=0.55,py0=0.0956,Tinf=286.15,Tp0 = 289.15,Y = UseRHCalcVaporMassFrac(0.48,286.15)
    #         ,pmfr=pmfr10,SRCoeff=SRCoeff,Type=Type)
    #     #1.6
    # =============================================================================
    calc_value1 = value(
        378.15, 1, 1, hw=0.65, py0=0.20, Tinf=286.15, Tp0=301.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_Final, Type=Final)
    calc_value1 = value(
        378.15, 1, 1, hw=0.65, py0=0.20, Tinf=286.15, Tp0=301.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_Type5, Type=Type5)


# =============================================================================
#     calc_value = calc_value - 0.2260
#     calc_value1 = calc_value1 - 0.4517
#     return calc_value**2 + calc_value1**2
# =============================================================================

def draw_pic(SRCoeff, Type):
    """
    对液滴喷雾干燥历程的可视化绘图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    from matplotlib.font_manager import FontProperties
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    legend_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    # SM10
    pxList0, srList0, pwList0, tbList0, tpList0, YList0, l0, Time0 = valueList(
        383.15, 1, 1.51, hw=0.55, py0=0.1, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr10, SRCoeff=SRCoeff_Final, Type=Final)
    pxList3, srList3, pwList3, tbList3, tpList3, YList3, l3, Time3 = valueList(
        383.15, 1, 1.51, hw=0.55, py0=0.1, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr10, SRCoeff=SRCoeff_Type5, Type=Type5)
    pxList6, srList6, pwList6, tbList6, tpList6, YList6, l6, Time6 = valueList(
        383.15, 1, 1.51, hw=0.55, py0=0.1, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr10, SRCoeff=SRCoeff_Type4, Type=Type4)
    pxList9, srList9, pwList9, tbList9, tpList9, YList9, l9, Time9 = valueList(
        383.15, 1, 1.51, hw=0.55, py0=0.1, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr10, SRCoeff=SRCoeff_Type3, Type=Type3)
    pxList12, srList12, pwList12, tbList12, tpList12, YList12, l12, Time12 = valueList(
        383.15, 1, 1.51, hw=0.55, py0=0.1, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr10, SRCoeff=SRCoeff_NanFu, Type=NanFu)
    # SM20
    pxList1, srList1, pwList1, tbList1, tpList1, YList1, l1, Time1 = valueList(
        383.15, 1, 1.51, hw=0.6, py0=0.198225, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr20, SRCoeff=SRCoeff_Final, Type=Final)
    pxList4, srList4, pwList4, tbList4, tpList4, YList4, l4, Time4 = valueList(
        383.15, 1, 1.51, hw=0.6, py0=0.198225, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr20, SRCoeff=SRCoeff_Type5, Type=Type5)
    pxList7, srList7, pwList7, tbList7, tpList7, YList7, l7, Time7 = valueList(
        383.15, 1, 1.51, hw=0.6, py0=0.198225, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr20, SRCoeff=SRCoeff_Type4, Type=Type4)
    pxList10, srList10, pwList10, tbList10, tpList10, YList10, l10, Time10 = valueList(
        383.15, 1, 1.51, hw=0.6, py0=0.198225, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr20, SRCoeff=SRCoeff_Type3, Type=Type3)
    pxList13, srList13, pwList13, tbList13, tpList13, YList13, l13, Time13 = valueList(
        383.15, 1, 1.51, hw=0.6, py0=0.198225, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr20, SRCoeff=SRCoeff_NanFu, Type=NanFu)
    # SM30
    pxList2, srList2, pwList2, tbList2, tpList2, YList2, l2, Time2 = valueList(
        383.15, 1, 1.51, hw=0.65, py0=0.29412181544369, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_Final, Type=Final)  # hw=0.65
    pxList5, srList5, pwList5, tbList5, tpList5, YList5, l5, Time5 = valueList(
        383.15, 1, 1.51, hw=0.65, py0=0.29412181544369, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_Type5, Type=Type5)  # hw=0.65
    pxList8, srList8, pwList8, tbList8, tpList8, YList8, l8, Time8 = valueList(
        383.15, 1, 1.51, hw=0.65, py0=0.29412181544369, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_Type4, Type=Type4)  # hw=0.65
    pxList11, srList11, pwList11, tbList11, tpList11, YList11, l11, Time11 = valueList(
        383.15, 1, 1.51, hw=0.65, py0=0.29412181544369, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_Type3, Type=Type3)  # hw=0.65
    pxList14, srList14, pwList14, tbList14, tpList14, YList14, l14, Time14 = valueList(
        383.15, 1, 1.51, hw=0.65, py0=0.29412181544369, Tinf=286.15, Tp0=298.15, Y=UseRHCalcVaporMassFrac(0.48, 286.15)
        , pmfr=pmfr30, SRCoeff=SRCoeff_NanFu, Type=NanFu)  # hw=0.65

    workbook = xlsxwriter.Workbook('NN0.xlsx')
    worksheet = workbook.add_worksheet('sheet1')

    worksheet.write_column('A2', Time0)
    worksheet.write_column('B2', l0)
    worksheet.write_column('C2', srList0)

    worksheet.write_column('E2', Time3)
    worksheet.write_column('F2', l3)
    worksheet.write_column('G2', srList3)

    worksheet.write_column('I2', Time6)
    worksheet.write_column('J2', l6)
    worksheet.write_column('K2', srList6)

    worksheet.write_column('M2', Time9)
    worksheet.write_column('N2', l9)
    worksheet.write_column('O2', srList9)

    worksheet.write_column('S2', Time12)
    worksheet.write_column('T2', l12)
    worksheet.write_column('U2', srList12)
    worksheet = workbook.add_worksheet('sheet2')

    worksheet.write_column('A2', Time1)
    worksheet.write_column('B2', l1)
    worksheet.write_column('C2', srList1)

    worksheet.write_column('E2', Time4)
    worksheet.write_column('F2', l4)
    worksheet.write_column('G2', srList4)

    worksheet.write_column('I2', Time7)
    worksheet.write_column('J2', l7)
    worksheet.write_column('K2', srList7)

    worksheet.write_column('M2', Time10)
    worksheet.write_column('N2', l10)
    worksheet.write_column('O2', srList10)

    worksheet.write_column('S2', Time13)
    worksheet.write_column('T2', l13)
    worksheet.write_column('U2', srList13)
    worksheet = workbook.add_worksheet('sheet3')

    worksheet.write_column('A2', Time2)
    worksheet.write_column('B2', l2)
    worksheet.write_column('C2', srList2)

    worksheet.write_column('E2', Time5)
    worksheet.write_column('F2', l5)
    worksheet.write_column('G2', srList5)

    worksheet.write_column('I2', Time8)
    worksheet.write_column('J2', l8)
    worksheet.write_column('K2', srList8)

    worksheet.write_column('M2', Time11)
    worksheet.write_column('N2', l11)
    worksheet.write_column('O2', srList11)

    worksheet.write_column('S2', Time14)
    worksheet.write_column('T2', l14)
    worksheet.write_column('U2', srList14)
    workbook.close()
    axes[0, 0].set_title("在塔内不同位置下脱脂奶液滴的温度", fontproperties=font_set)
    axes[0, 0].set_xlabel(u"到塔顶的距离(m)", fontproperties=font_set)
    axes[0, 0].set_ylabel(u"脱脂奶液滴的温度(℃)", fontproperties=font_set)
    axes[0, 0].plot(l0, tbList0, label=r'10%RSM热空气温度')
    axes[0, 0].plot(l1, tbList1, label=r'20%RSM热空气温度')
    axes[0, 0].plot(l2, tbList2, label=r'30%RSM热空气温度')
    axes[0, 0].plot(l0, tpList0, label=r'10%RSM颗粒温度')
    axes[0, 0].plot(l1, tpList1, label=r'20%RSM颗粒温度')
    axes[0, 0].plot(l2, tpList2, label=r'30%RSM颗粒温度')
    axes[0, 0].set_xlim([0, 3.1])
    axes[0, 0].legend(loc='best', prop=legend_set)

    axes[1, 0].set_title("在塔内不同位置下脱脂奶液滴的温度", fontproperties=font_set)
    axes[1, 0].set_xlabel(u"停留时间(s)", fontproperties=font_set)
    axes[1, 0].set_ylabel(u"脱脂奶液滴的温度(℃)", fontproperties=font_set)
    axes[1, 0].plot(Time0, tbList0, label=r'10%RSM热空气温度')
    axes[1, 0].plot(Time1, tbList1, label=r'20%RSM热空气温度')
    axes[1, 0].plot(Time2, tbList2, label=r'30%RSM热空气温度')
    axes[1, 0].plot(Time0, tpList0, label=r'10%RSM颗粒温度')
    axes[1, 0].plot(Time1, tpList1, label=r'20%RSM颗粒温度')
    axes[1, 0].plot(Time2, tpList2, label=r'30%RSM颗粒温度')
    axes[1, 0].set_xlim([0, 15])
    axes[1, 0].legend(loc='best', prop=legend_set)

    axes[0, 1].set_title("在塔内不同位置下脱脂奶液滴的含水率", fontproperties=font_set)
    axes[0, 1].set_xlabel(u"停留时间(s)", fontproperties=font_set)
    axes[0, 1].set_ylabel(u"干基含水率", fontproperties=font_set)
    axes[0, 1].plot(l0, pxList0, label=r'10%RSM')
    axes[0, 1].plot(l1, pxList1, label=r'20%RSM')
    axes[0, 1].plot(l2, pxList2, label=r'30%RSM')
    axes[0, 1].scatter(3.0, 0.121398352156732, s=15, label=r'10%RSM')
    axes[0, 1].scatter(3.0, 0.0935528960287667, s=15, label=r'20%RSM')
    axes[0, 1].scatter(3.0, 0.0854195207352735, s=15, label=r'30%RSM')
    axes[0, 1].set_xlim([0, 3.1])
    axes[0, 1].set_ylim([0, 9])
    axes[0, 1].legend(loc='best', prop=legend_set)

    axes[1, 1].set_title("在塔内不同位置下脱脂奶液滴的含水率", fontproperties=font_set)
    axes[1, 1].set_xlabel(u"到塔顶的距离(m)", fontproperties=font_set)
    axes[1, 1].set_ylabel(u"干基含水率", fontproperties=font_set)
    axes[1, 1].plot(Time0, pxList0, label=r'10%RSM')
    axes[1, 1].plot(Time1, pxList1, label=r'20%RSM')
    axes[1, 1].plot(Time2, pxList2, label=r'30%RSM')
    axes[1, 1].scatter(Time0[-1], 0.121398352156732, s=15, label=r'10%RSM')
    axes[1, 1].scatter(Time1[-1], 0.0935528960287667, s=15, label=r'20%RSM')
    axes[1, 1].scatter(Time2[-1], 0.0854195207352735, s=15, label=r'30%RSM')
    axes[1, 1].set_xlim([0, 15])
    axes[1, 1].set_ylim([0, 9])
    axes[1, 1].legend(loc='best', prop=legend_set)

    # axes[0, 2].set_title("在塔内不同位置下细菌存活率", fontproperties=font_set)
    # axes[0, 2].set_xlabel(u"到塔顶的距离(m)", fontproperties=font_set)
    # axes[0, 2].set_ylabel(u"益生菌的存活率", fontproperties=font_set)
    # axes[0, 2].plot(l0, srList0, label=r'10%RSM_TXdTdX')
    # axes[0, 2].plot(l1, srList1, label=r'20%RSM_TXdTdX')
    # axes[0, 2].plot(l2, srList2, label=r'30%RSM_TXdTdX')
    # axes[0, 2].plot(l3, srList3, label=r'10%RSM_TXdX')
    # axes[0, 2].plot(l4, srList4, label=r'20%RSM_TXdX')
    # axes[0, 2].plot(l5, srList5, label=r'30%RSM_TXdX')
    # axes[0, 2].plot(l6, srList6, label=r'10%RSM_TdX')
    # axes[0, 2].plot(l7, srList7, label=r'20%RSM_TdX')
    # axes[0, 2].plot(l8, srList8, label=r'30%RSM_TdX')
    # axes[0, 2].plot(l9, srList9, label=r'10%RSM_TX')
    # axes[0, 2].plot(l10, srList10, label=r'20%RSM_TX')
    # axes[0, 2].plot(l11, srList11, label=r'30%RSM_TX')
    # axes[0, 2].plot(l12, srList12, label=r'10%RSM_T')
    # axes[0, 2].plot(l13, srList13, label=r'20%RSM_T')
    # axes[0, 2].plot(l14, srList14, label=r'30%RSM_T')
    # axes[0, 2].set_xlim([0, 3.1])
    # axes[0, 2].scatter(3, 0.23549178265568, s=15, label=r'10%RSM experiment')
    # axes[0, 2].scatter(3, 0.408611317334794, s=15, label=r'20%RSM experiment')
    # axes[0, 2].scatter(3, 0.326394979194388, s=15, label=r'30%RSM experiment')
    # axes[0, 2].yaxis.set_major_formatter(formatter)
    # axes[0, 2].legend(loc=10, prop=legend_set)

    # =============================================================================
    #     axes[1, 2].set_title("在塔内不同位置下空气的的相对湿度", fontproperties=font_set)
    #     axes[1, 2].set_xlabel(u"到塔顶的距离(m)", fontproperties=font_set)
    #     axes[1, 2].set_ylabel(u"空气的相对湿度(%)", fontproperties=font_set)
    #     axes[1, 2].plot(l0, np.array(list(map(UseVaporMassFracCalcRH,zip(YList0,tbList0))))*100, label=r'10%RSM')
    #     axes[1, 2].plot(l, np.array(list(map(UseVaporMassFracCalcRH,zip(YList,tbList))))*100, label=r'30%RSM')
    #     axes[1, 2].set_xlim([0,3.1])
    #     axes[1, 2].legend(loc=0,prop=legend_set)
    # =============================================================================

    # axes[1, 2].set_title("益生菌的存活率与停留时间", fontproperties=font_set)
    # axes[1, 2].set_xlabel(u"停留时间(s)", fontproperties=font_set)
    # axes[1, 2].set_ylabel(u"益生菌的存活率", fontproperties=font_set)
    # axes[1, 2].plot(Time0, srList0, label=r'10%RSM_TXdTdX')
    # axes[1, 2].plot(Time1, srList1, label=r'10%RSM_TXdTdX')
    # axes[1, 2].plot(Time2, srList2, label=r'30%RSM_TXdTdX')
    # axes[1, 2].plot(Time3, srList3, label=r'10%RSM_TXdX')
    # axes[1, 2].plot(Time4, srList4, label=r'10%RSM_TXdX')
    # axes[1, 2].plot(Time5, srList5, label=r'30%RSM_TXdX')
    # axes[1, 2].plot(Time6, srList6, label=r'10%RSM_TdX')
    # axes[1, 2].plot(Time7, srList7, label=r'10%RSM_TdX')
    # axes[1, 2].plot(Time8, srList8, label=r'30%RSM_TdX')
    # axes[1, 2].plot(Time9, srList9, label=r'10%RSM_TX')
    # axes[1, 2].plot(Time10, srList10, label=r'10%RSM_TX')
    # axes[1, 2].plot(Time11, srList11, label=r'30%RSM_TX')
    # axes[1, 2].plot(Time12, srList12, label=r'10%RSM_T')
    # axes[1, 2].plot(Time13, srList13, label=r'10%RSM_T')
    # axes[1, 2].plot(Time14, srList14, label=r'30%RSM_T')
    # axes[1, 2].scatter(Time0[-1], 0.23549178265568, s=15, label=r'10%RSM experiment')
    # axes[1, 2].scatter(Time1[-1], 0.408611317334794, s=15, label=r'20%RSM experiment')
    # axes[1, 2].scatter(Time2[-1], 0.326394979194388, s=15, label=r'30%RSM experiment')
    # axes[1, 2].set_xlim([0,15])
    # axes[1, 2].yaxis.set_major_formatter(formatter)
    # axes[1, 2].legend(loc=0, prop=legend_set)

    # =============================================================================
    #     线性回归
    # =============================================================================

    # =============================================================================
    #     test = pd.DataFrame([Time0, pxList0, tpList0])
    #     test.index = ["Time","X","T"]
    #     test["dXdt"] = test["X"][1:]/test["Time"].diff()
    #     test["dTdt"] = test["X"][1:]/test["Time"].diff()
    #
    #     test = test.dropna()
    # =============================================================================
    # =============================================================================
    #     test = test.loc[test.loc[:,'lnkd']>-1000,:]
    #     test = test.loc[test.loc[:,'N/N0']<1,:]
    #     test['1/T'] = 1./(test['T']+273.15)*1000
    #     test['1/T2'] = test['1/T']**2
    #     test['dX/dt2'] = test['dX/dt']**2
    #     plt.scatter(test['1/T'],test['lnkd'])
    #     Y = test['lnkd']
    #     X = test.loc[:,['1/T']] #'X',['X','1/T','dT/dt','dX/dt']
    #     X = np.array(X)
    #     plt.scatter(test['1/T'],test['lnkd'])
    #     x = np.arange(2.5,3.5,0.1)y = 33.28+(-12.7)*x
    #     plt.plot(x,y)
    #     x = np.arange(2.5,3.5,0.1)y = 16.3 + (-7)*x
    #     plt.plot(x,y)
    #     reg = linear_model.LinearRegression()
    #     reg.fit(X,Y)
    #     print(reg.coef_)
    #     print(reg.intercept_)
    #     print(reg.score(X,Y))
    # =============================================================================
    # =============================================================================
    #
    # =============================================================================

    # plt.suptitle("脱脂奶液滴在喷雾干燥过程中计算与实验对比", fontproperties=font_set_title, color='red')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.35)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    sns.despine()
    plt.show()


if __name__ == "__main__":
    # fitness = diffWithExp(SRCoeff_Type5,Type=Type5)
    # fitness = diffWithExp(SRCoeff_Type4,Type=Type4)
    # fitness = diffWithExp(SRCoeff_Final,Type=Final)
    # print(diffWithExp(SRCoeff_Final,Type=Final))
    # draw_pic(SRCoeff_NanFu,NanFu)
    # draw_pic(SRCoeff_Type5,Type5)
    # draw_pic(SRCoeff_Type4,Type4)
    # draw_pic(SRCoeff_Type3,Type3)
    # draw_pic(SRCoeff_Regression,Regression)
    draw_pic(SRCoeff_Regression, Regression)

# =============================================================================
#     rhoDrop0 = (1 - 0.2)*calcWaterRho(298.15) + 0.2*calcMilkRho(298.15)
#     pMass0 = 2.24e-4 ** 3 * pi / 6. * rhoDrop0
#     ms = pMass0 * 0.2
# =============================================================================
