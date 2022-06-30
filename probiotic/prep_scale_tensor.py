import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import pickle
import numpy as np
import probioticsCCE.my_functions as mf
from matplotlib import pyplot as plt

len_train_val, len_test = [], []  # record the length of each df for the following split
ft_train_val, ft_test = [], []  # store all the ft for rescaling
s_train_val, s_test = [], []
test_id = [3, 5]  # group 3 and group 5 constitute the test set
lgs = False
s_tilde = False

for i in range(8):
    df = pd.read_excel('excel/itp_ft_s.xlsx', sheet_name='Sheet' + str(i + 1))
    df['tag'] = df['tag'].astype(int)
    ft = df.iloc[:, :3].values
    s = df.iloc[:, 5:].values
    if i + 1 in test_id:
        len_test.append(df.shape[0])
        ft_test.append(ft)
        s_test.append(s)
    else:
        len_train_val.append(df.shape[0])
        ft_train_val.append(ft)
        s_train_val.append(s)
# The four features should be rescaled (StandardScalar or MinMaxScalar)
ft_train_val, ft_test = np.vstack(ft_train_val), np.vstack(ft_test)
ft_scalar = StandardScaler()
# use the scalar fitted with the train_val set to fit the test set
ft_train_val_scaled = torch.from_numpy(ft_scalar.fit_transform(ft_train_val).astype(np.float32))
ft_test_scaled = torch.from_numpy(ft_scalar.transform(ft_test).astype(np.float32))

s_train_val, s_test = np.vstack(s_train_val), np.vstack(s_test)
plt.figure(1)
plt.hist(np.vstack((s_train_val[:, [0]], s_test[:, [0]])), bins=600)
plt.xlabel('Original s', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.yscale('log')
s_tv_value, s_tv_tag = s_train_val[:, [0]], s_train_val[:, [1]]
s_test_value, s_test_tag = s_test[:, [0]], s_test[:, [1]]
if lgs:
    s_tv_value = np.log10(s_tv_value)
    s_test_value = np.log10(s_test_value)
s_scalar = None
if s_tilde:
    s_scalar = StandardScaler()
    s_scalar.fit(s_tv_value)
    s_tv_value = s_scalar.transform(s_tv_value)
    s_test_value = s_scalar.transform(s_test_value)
plt.figure(2)
plt.hist(np.vstack((s_tv_value, s_test_value)), bins=600)
plt.xlabel('Preprocessed s', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.yscale('log')

s_train_val_scaled = torch.from_numpy(np.hstack((s_tv_value, s_tv_tag)).astype(np.float32))
s_test_scaled = torch.from_numpy(np.hstack((s_test_value, s_test_tag)).astype(np.float32))

# concat ft and s for rnn.pad_sequence
ft_s_tv = torch.hstack((ft_train_val_scaled, s_train_val_scaled))
ft_s_test = torch.hstack((ft_test_scaled, s_test_scaled))
# split the sets
len_sum_tv = np.r_[0, np.cumsum(len_train_val)]
len_sum_test = np.r_[0, np.cumsum(len_test)]
ft_s_tv = [ft_s_tv[len_sum_tv[i]:len_sum_tv[i + 1], :] for i in range(6)]
ft_s_test = [ft_s_test[len_sum_test[i]:len_sum_test[i + 1], :] for i in range(2)]


# storage
with open('pickle_data/test35_s_non-tilde.pkl', 'wb') as pf1:
    pickle.dump({
        'ft_s_tv': ft_s_tv, 'ft_s_test': ft_s_test,
        'ft_scalar': ft_scalar, 's_scalar': s_scalar
    }, pf1)
