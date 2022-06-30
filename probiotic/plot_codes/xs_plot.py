import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# see how water content (X) and survival rate (s) are correlated
xs = []
for i in range(8):
    df = pd.read_excel('itp_ft_s.xlsx', sheet_name='Sheet' + str(i + 1))
    xs.append(df.loc[:, ['t(s)', 'X', 'itp']].to_numpy())
    plt.scatter(xs[-1][:, 1], xs[-1][:, 2], s=3, label=str(i + 1))
plt.legend(loc='best')
plt.xlabel('t (s)')
plt.ylabel('survival rate')
