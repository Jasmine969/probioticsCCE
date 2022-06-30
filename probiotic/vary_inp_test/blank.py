import numpy as np
from pred_fun import vary_pred_plot

elapse = 360
interval = 1
t = np.arange(0, elapse, interval)
T = np.ones_like(t) * 290
X = np.ones_like(t) * 9
# X = np.linspace(9,-50,n)
vary_pred(g=0, t_scales=[1], t=t, temp=T,
          moisture=X, cfu_tag=False, ylim=[0, 1.1], ops='win')
