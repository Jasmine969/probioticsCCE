import pickle
from scipy import io as sio
import numpy as np
import pandas as pd

with open('../pickle_data/test35_s_non-tilde.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    ft_s_tv = dct['ft_s_tv']
    ft_scalar = dct['ft_scalar']
with open('../pickle_data/test35_lgg_s.pkl', 'rb') as pf2:
    ft_s_test = pickle.load(pf2)

n_tv = [1, 2, 4, 6, 7, 8]


def func(ori_data):
    fts_reg, kds_reg = [], []
    fts_rst, ss_obs = [], []  # rst--restore
    tags = []
    for i in range(len(ori_data)):
        data = ori_data[i].numpy()
        data[:, :3] = ft_scalar.inverse_transform(data[:, :3])
        t, x, y, tag = np.split(data, [1, 3, 4], axis=1)
        tag = tag.astype(bool)
        t = (t[tag]).round().astype(int)
        diff_x = np.abs(np.diff(x, axis=0))
        # test data at all time-points
        ft_test = np.hstack((x[1:], diff_x))
        s_obs = y[t]
        # regression data at sampling time-point
        if len(ori_data) == 6:  # tv
            kd_data = pd.read_excel('kd.xlsx', sheet_name='G' + str(n_tv[i])).to_numpy()
            t_reg = kd_data[:, 0].astype(int)
            ft_reg = ft_test[t_reg - 1]
            kd_reg = kd_data[:, 1]
            print(ft_reg.shape[0])

        else:  # test
            kd_reg = []
            ft_reg = []
        # save
        fts_reg.append(ft_reg)
        kds_reg.append(kd_reg)
        fts_rst.append(ft_test.T)
        ss_obs.append(s_obs.flatten())
        tags.append(tag.flatten())
    fts_reg = np.vstack(fts_reg)
    kds_reg = np.hstack(kds_reg)
    kds_reg = np.log(kds_reg)  # ln(kd) at sampling time-points)
    return fts_reg.T, kds_reg, fts_rst, ss_obs, tags


x_reg, y_reg, x_rst, s_gt, tag_rst = func(ft_s_tv)
_, _, x_test, s_test, tag_test = func(ft_s_test)
t_rst = [np.where(tag_rs)[0].astype(float) for tag_rs in tag_rst]
t_test = [np.where(tag_tes)[0].astype(float) for tag_tes in tag_test]
dct = {
    'x_reg': x_reg, 'y_reg': y_reg,
    'x_rst': x_rst, 's_gt': s_gt,
    'x_test': x_test, 's_test': s_test,
    'tag_rst': tag_rst, 'tag_test': tag_test,
    't_rst': t_rst, 't_test': t_test
}
with open('../pickle_data/for_other_models_dataset.pkl', 'wb') as pf:
    pickle.dump(dct, pf)
sio.savemat('other_model_dataset.mat', {'result': {
    # 'x_reg': x_reg, 'y_reg': y_reg,
    # 'x_rst': x_rst,
    's_gt': s_gt,
    # 'x_test': x_test,
    's_test': s_test,
    # 'tag_rst': tag_rst, 'tag_test': tag_test
    't_rst': t_rst, 't_test': t_test
}})
