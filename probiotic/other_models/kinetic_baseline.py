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
k = mf.least_square(data[:, [0]], data[:, [1]])
pred_dsdt = k * data[:,[0]]
plt.scatter(data[:,[1]], pred_dsdt)
print(f'r2={r2_score(data[:,[1]],pred_dsdt)}')
plt.show()