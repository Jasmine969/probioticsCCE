import pandas as pd
import numpy as np

writer = pd.ExcelWriter('excel\\raw_1s.xlsx')
# 前六组
params6 = {'io': 'excel\\RawG1-G6.xlsx', 'sheet_name': 'Sheet5', 'header': 1}
cols6 = ['AB:AF', 'AK:AO', 'AS:AW', 'D:H', 'L:P', 'T:X']
rows6 = [270, 270, 120, 300, 300, 120]
for i in range(6):
    col_names = iter(['t(s)', 'T(K)', 'X', 'dTdt', 'dXdt'])
    df6 = pd.read_excel(**params6, usecols=cols6[i], nrows=rows6[i] + 1)
    df6.rename(columns=lambda x: next(col_names), inplace=True)
    # df6['dXdt'] = -df6['dXdt']  # moisture removal rate
    df6.to_excel(writer, index=None, sheet_name='Sheet' + str(i + 1))

# 后两组
file_name2 = 'excel/RawG7-G8.xlsx'
sheet_name2 = ['1ul', '2ul']
for i in range(7, 9):
    df2 = pd.read_excel(file_name2, sheet_name=sheet_name2[i - 7], usecols='A:C,M', nrows=402, header=3)
    # moisture removal rate
    df2['dXdt'] = np.r_[df2.iloc[1:, 3].to_numpy() - df2.iloc[:-1, 3].to_numpy(), np.nan]
    df2.drop(401, inplace=True)
    col_ori_names = df2.columns.to_list()
    df2[[col_ori_names[2], col_ori_names[3]]] = df2[[col_ori_names[3], col_ori_names[2]]]
    col_names = iter(['t(s)', 'T(K)', 'X', 'dTdt', 'dXdt'])
    df2.rename(columns=lambda x: next(col_names), inplace=True)
    df2['T(K)'] = df2['T(K)'] + 273.15
    df2.to_excel(writer, index=None, sheet_name='Sheet' + str(i))

writer.save()
