import pandas as pd
from matplotlib import pyplot as plt

# 主要进行merge操作
file_ft = 'excel\\raw_1s.xlsx'
file_s = 'excel\\1_乳酸菌 数据.xlsx'
headers = [4, 18, 32, 41, 56, 71, 80, 92]
nrows = [12, 12, 7, 13, 13, 7, 10, 10]
writer = pd.ExcelWriter('excel\\raw_ft_s.xlsx')
for i in range(8):
    df_ft = pd.read_excel(file_ft, sheet_name='Sheet' + str(i + 1))
    df_s = pd.read_excel(file_s, sheet_name='Sheet2',
                         header=headers[i], nrows=nrows[i], usecols='I,D')
    df_s['c'].clip(0, 1, inplace=True)
    # # compare the length of {T} and {s}
    # if i == 0:
    #     fig, ax1 = plt.subplots()
    #     ax2 = ax1.twinx()
    #     ax1.set_ylabel('T (K)',fontsize=14)
    #     ax1.set_xlabel('Time (s)',fontsize=14)
    #     ax1.plot(df_ft['t(s)'].to_numpy(), df_ft['T(K)'].to_numpy(), '.-',color='orange',label='T')
    #     ax2.set_ylabel('survival rate',fontsize=14)
    #     plt.plot(df_s['t(s)'].to_numpy(), df_s['c'].to_numpy(), '.',markersize=10, label='s')
    #     plt.legend(loc='best')
    #     handles1, labels1 = ax1.get_legend_handles_labels()
    #     handles2, labels2 = ax2.get_legend_handles_labels()
    #     plt.legend(handles1 + handles2, labels1 + labels2, loc='center left')
    df_ft = pd.merge(df_ft, df_s, how='inner', on='t(s)')
    columns = df_ft.columns.to_list()
    columns[-1] = 's'
    columns = iter(columns)
    df_ft.rename(columns=lambda x: next(columns), inplace=True)
    # # visualize clamping
    # if i > 5:
    #     plt.figure(i + 1)
    #     plt.plot(df_ft['t(s)'].to_numpy(), df_ft['s'].to_numpy(), '--', label='before')
    #     df_ft['s'].clip(0., 1., inplace=True)  # clamp the survival rate between 0 and 1
    #     plt.plot(df_ft['t(s)'].to_numpy(), df_ft['s'].to_numpy(), label='after')
    #     plt.legend(loc='best')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('survival rate')
    #     plt.title(F'Group {i + 1}')
    df_ft.to_excel(writer, index=None, sheet_name='Sheet' + str(i + 1))
writer.save()
