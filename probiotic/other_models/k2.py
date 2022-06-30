import numpy as np
import pickle
from scipy import optimize
from sklearn.metrics import r2_score
import probioticsCCE.my_functions as mf
from matplotlib import pyplot as plt

with open('../pickle_data/for_other_models_dataset.pkl', 'rb') as pf:
    dct = pickle.load(pf)
    x_reg = dct['x_reg']
    y_reg = dct['y_reg']
    x_rst = dct['x_rst']
    s_gt = dct['s_gt']
    x_test = dct['x_test']
    s_test = dct['s_test']
    tag_rst = dct['tag_rst']
    tag_test = dct['tag_test']
r = 8.314

ta_reg = [70] * 8 + [90] * 4 + [70] * 6 + [
    110] * 1 + [70] * 9 + [70] * 7
ta_reg = np.asarray(ta_reg).reshape(1, -1).astype(np.float)
ta_rst = [[70] * 270, [90] * 270, [70] * 300, [
    110] * 120, [70] * 400, [70] * 400]
ta_rst = [np.asarray(i).reshape(1, -1).astype(np.float) for i in ta_rst]
# ta_rst = np.asarray(ta_rst).reshape(1, -1).astype(np.float)
ta_test = [np.ones_like(
    group[[1], :]) * ta for group, ta in zip(x_test, [110, 90, 70])]
pass

def mylog(x):
    return np.log(np.clip(x, 1e-40, 1e10))


def fun1(x, lnk0, ed):
    return lnk0 - ed / (r * x[0])


def fun2(x, lnk0, ed):
    return lnk0 - ed / (r * x[0])


def fun3(x, lnk0, a, b, ed):
    return lnk0 + a * x[1] - (ed + b * x[1]) / (r * x[0])


def fun4(x, lnk0, a, ed):
    return lnk0 + mylog(1 + a * x[3]) - ed / (r * x[0])


def fun5(x, lnk0, a, b, ed):
    return lnk0 + mylog(1 + a * x[2]) + mylog(1 + b * x[3]) - ed / (r * x[0])


def fun6(x, lnk0, a, b, ed):
    return lnk0 + mylog(1 + a * x[3] + b * x[3] ** 2) - ed / (r * x[0])


def fun7(x, lnk0, a, b, ed):
    return lnk0 + a * x[1] - (ed + b * x[0]) / (r * x[0])


def fun8(x, lnk0, a, b, ed):
    return lnk0 + a * x[0] - (ed + b * x[1]) / (r * x[0])


def fun9(x, lnk0, a, b, ed):
    return lnk0 + mylog(1 + a * x[2]) + b * x[1] - ed / (r * x[1])


# # Model 2-9
# fun = fun2
# Model 1
fun = fun1
x_reg = ta_reg
x_rst = ta_rst
x_test = ta_test
# p0 = [30, 1, 1e5]
# ind = np.argwhere(y_reg > -7)
# x_reg = x_reg[:, ind][:, :, 0]
# y_reg = y_reg[ind].flatten()
popt, _ = optimize.curve_fit(fun, x_reg, y_reg)
print(f'ori_r2: {r2_score(y_reg, fun(x_reg, *popt)):.4f}')
# fit acc
r2_fit_lnr, r2_fit_lg = [], []
for i in range(6):
    pred_kd = fun(x_rst[i], *popt)
    pred = mf.kd2s(np.exp(pred_kd))
    r2_lnr = r2_score(s_gt[i], pred[tag_rst[i]])
    r2_fit_lnr.append(r2_lnr)
    # pred_lg = np.log10(np.clip(pred, 1e-7, 10))
    pred_lg = np.log10(pred)
    r2_lg = r2_score(np.log10(s_gt[i]), pred_lg[tag_rst[i]])
    r2_fit_lg.append(r2_lg)
# test acc
r2_test_lnr, r2_test_lg = [], []
for i in range(3):
    pred_kd = fun(x_test[i], *popt)
    pred = mf.kd2s(np.exp(pred_kd))
    r2_lnr = r2_score(s_test[i], pred[tag_test[i]])
    r2_test_lnr.append(r2_lnr)
    pred_lg = np.log10(np.clip(pred, 1e-7, 10))
    r2_lg = r2_score(np.log10(s_test[i]), pred_lg[tag_test[i]])
    r2_test_lg.append(r2_lg)
fit_lnr_acc, fit_lg_acc = sum(r2_fit_lnr) / 6, sum(r2_fit_lg) / 6
test_lnr_acc, test_lg_acc = sum(r2_test_lnr) / 3, sum(r2_test_lg) / 3
print(popt)
print(f'k_0={np.exp(popt[0]):.3e}')
# print(r2_fit_lg)
print(f'fit_lnr_acc: {fit_lnr_acc:.4f}')
print(f'fit_lg_acc: {fit_lg_acc:.4f}')
print(f'test_lnr_acc: {test_lnr_acc:.4f}')
print(f'test_lg_acc: {test_lg_acc:.4f}')
