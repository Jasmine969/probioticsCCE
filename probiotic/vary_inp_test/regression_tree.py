import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import pickle
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.metrics import r2_score
from dtreeviz.trees import dtreeviz

with open('../pickle_data/rate5x5x7x5.pkl', 'rb') as file:
    exp_cond, rates = pickle.load(file)

ta_tuple, va_tuple, ws_tuple, vd_tuple = zip(*exp_cond)
exp_cond_ar = np.array([ta_tuple, va_tuple, ws_tuple, vd_tuple])
exp_cond_ar[0, :] -= 273.15
exp_cond_ar[3, :] *= 1e9
exp_cond_ar = exp_cond_ar.T
rates = np.array(rates) * 1000
# Data set

# Fit regression model
model1 = DecisionTreeRegressor(max_depth=7, criterion='mse')
model1.fit(exp_cond_ar, rates)

# Predict
rates_pred1 = model1.predict(exp_cond_ar)

# Plot the results
plt.figure()
plt.scatter(rates, rates_pred1)
# plt.plot([-0.01, -0.001], [-0.01, -0.001], 'r')
# plt.plot(exp_cond_ar[:, 1], rates_pred1, color="yellowgreen", label="max_depth=7", linewidth=2)
# plt.plot(exp_cond_ar[:, 2], rates_pred1, color="yellowgreen", label="max_depth=7", linewidth=2)
# plt.plot(exp_cond_ar[:, 3], rates_pred1, color="yellowgreen", label="max_depth=7", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title(f'r2={r2_score(rates, rates_pred1):.4f}')
plt.show()
# dot_data = tree.export_graphviz(model1, out_file=None,
#                                 feature_names=['Ta', 'va', 'ws', 'Vd'],  # 对应特征的名字
#                                 class_names=['y'],  # 对应类别的名字
#                                 filled=True, rounded=True,
#                                 special_characters=True, precision=5,max_depth=4)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('tree.png')
viz = dtreeviz(
    model1,
    exp_cond_ar,
    rates,
    target_name='rate'
)