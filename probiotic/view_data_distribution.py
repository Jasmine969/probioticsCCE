import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

s = []
for i in range(8):
    df = pd.read_excel('excel/itp_ft_s.xlsx', sheet_name='Sheet' + str(i + 1))
    s.extend(df.iloc[:, 5].to_list())
s = np.asarray(s) - 1e-5
s_new = np.log(s / (1 - s))
plt.hist(s, bins=64)
plt.figure(2)
plt.hist(s_new, bins=64)
plt.figure(3)
plt.hist(np.log(s_new),bins=64)
plt.show()
