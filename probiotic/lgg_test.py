import pandas as pd
import pickle
import torch
from gen_bag_model import gen_bag
import my_functions as mf
from torch.nn.utils import rnn
from matplotlib import pyplot as plt, font_manager as fm
import numpy as np
from sklearn.metrics import r2_score
from scipy.interpolate import PchipInterpolator as phicp

kw_ft = pd.read_excel('excel/read_lgg.xlsx', 'Sheet1')
kw_s = pd.read_excel('excel/read_lgg.xlsx', 'Sheet2')
font_text = {'size': 17}
font_formula = fm.FontProperties(
    math_fontfamily='cm', size=19)
plt.rc('font', family='Times New Roman', size=13)
g = 4
ft = pd.read_excel(
    'excel/LGG.xlsx', **kw_ft.loc[g - 1].to_dict()
).to_numpy()
t_s = pd.read_excel(
    'excel/LGG.xlsx', **kw_s.loc[g - 1].to_dict()
).to_numpy()
with open('pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
    dct = pickle.load(pf)
    ft_scalar = dct['ft_scalar']
ft[:, 1] = ft[:, 1] + 273.15
tt = ft[:, 0]
s = t_s[:, 1]
t = t_s[:, 0]
# length = [ft.shape[0]]
real_ind = np.zeros(ft.shape[0]).astype(bool)
real_ind[(t.astype(int),)] = True
s = mf.mono_decrease(s)
interp = phicp(t, s)
ss = interp(tt)
ft_s_lgg = np.hstack((
    ft_scalar.transform(ft),
    ss[:, np.newaxis], real_ind[:, np.newaxis]
))
ft_s_lgg = torch.from_numpy(ft_s_lgg).type(torch.FloatTensor)
ft_s_test1 = dct['ft_s_test'] + [ft_s_lgg]
with open('pickle_data/test35_lgg_s.pkl','wb') as pf1:
    pickle.dump(ft_s_test1, pf1)
# model = gen_bag(ops='win')
# with torch.no_grad():
#     ft = rnn.pad_sequence([ft], batch_first=True).cuda()
#     pred, _ = model(ft, length)
# pred = mf.human_intervene(
#     pred, 1,
#     torch.from_numpy(real_ind), 0
# ).squeeze().cpu().numpy()
# pred_points = pred[real_ind]
#
# fig1 = plt.figure(1)
# plt.plot(t, pred, label='pred curve')
# plt.scatter(
#     s[:, 0], pred_points,
#     label='pred points', marker='v'
# )
# plt.scatter(
#     s[:, 0], s[:, 1],
#     marker='o', facecolor='white',
#     color='g', label='ground truth'
# )
# r2 = r2_score(s[:, 1], pred_points)
# plt.title(f'$R^2={r2:.4f}$')
# plt.legend(loc='best')
# plt.xlabel('Time (s)', fontdict=font_text)
# plt.ylabel(r'$s$', fontproperties=font_formula)
#
# fig2 = plt.figure(2)
# pred = np.log10(pred)
# pred_points = pred[real_ind]
# s[:, 1] = np.log10(s[:, 1])
# plt.plot(t, pred, label='pred curve')
# plt.scatter(
#     s[:, 0], pred_points,
#     label='pred points', marker='v'
# )
# plt.scatter(
#     s[:, 0], s[:, 1],
#     marker='o', facecolor='white',
#     color='g', label='ground truth'
# )
# r2 = r2_score(s[:, 1], pred_points)
# plt.title(f'$R^2={r2:.4f}$')
# plt.legend(loc='best')
# plt.xlabel('Time (s)', fontdict=font_text)
# plt.ylabel(r'$\lg s$', fontproperties=font_formula)
# plt.show()
