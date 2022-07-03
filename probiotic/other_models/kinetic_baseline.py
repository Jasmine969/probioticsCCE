import pandas as pd
import numpy as np
import pickle
import my_functions as mf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

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
data = [pd.read_excel('dsdt.xlsx', sheet_name='G' + str(
    i)) for i in [1, 2, 4, 6, 7, 8]]
data = np.vstack(data)
# least squared method
k = -mf.least_square(data[:, [0]], data[:, [1]])
# fit
pred_s_fit, pred_lgs_fit = [], []
r2_fit_lnr, r2_fit_lg = [], []
for i in range(6):
    time_fit = (np.where(tag_rst[i])[0]).astype(float)
    pred_lgs_fit.append(- k * time_fit / 2.303)
    pred_s_fit.append(10 ** pred_lgs_fit[-1])
    r2_fit_lnr.append(r2_score(s_gt[i], pred_s_fit[-1]))
    r2_fit_lg.append(r2_score(np.log10(s_gt[i]), pred_lgs_fit[-1]))
# test
pred_s_test, pred_lgs_test = [], []
r2_test_lnr, r2_test_lg = [], []
for i in range(3):
    time_test = (np.where(tag_test[i])[0]).astype(float)
    pred_lgs_test.append(- k * time_test / 2.303)
    pred_s_test.append(10 ** pred_lgs_test[-1])
    r2_test_lnr.append(r2_score(s_test[i], pred_s_test[-1]))
    r2_test_lg.append(r2_score(np.log10(s_test[i]), pred_lgs_test[-1]))
print(f'k: {k}')
print(f'r2_fit_lnr: {np.mean(r2_fit_lnr)}')
print(f'r2_fit_lg: {np.mean(r2_fit_lg)}')
print(f'r2_test_lnr: {np.mean(r2_test_lnr)}')
print(f'r2_test_lg: {np.mean(r2_test_lg)}')
