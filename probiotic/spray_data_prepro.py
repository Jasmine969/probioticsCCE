import pickle
import pandas as pd
import numpy as np
import torch

with open('pickle_data/test35_s_non-tilde.pkl', 'rb') as pf:
    dct = pickle.load(pf)
    ft_scalar = dct['ft_scalar']
data = pd.read_excel('excel/spray_extract_red.xlsx').to_numpy()
data = ft_scalar.transform(data)
data = torch.from_numpy(data).type(torch.FloatTensor)
s = 0.2355
with open('pickle_data/spray_red.pkl', 'wb') as pf1:
    pickle.dump({'ft': data, 's': s}, pf1)
