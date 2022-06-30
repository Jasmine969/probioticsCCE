import torch
from probiotic.MyNN import ConvAttn2Tower, BaggingModel


def gen_bag(ops):
    if ops == 'linux':
        parent_path = '/data/zhuhong_codes/python_code/probioticsCCE/probiotic/'
        path = parent_path + 'optuna_res/test35/'
    else:
        parent_path = 'E:/probioticsCCE/probiotic/'
        path = parent_path + 'optuna_res/test35/'
    # vali1
    model1 = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=79, dv=49,
        num_heads=6, dropout=0.399,
        act_name='leakyrelu'
    ).cuda()
    model1.load_state_dict(
        torch.load(path + 'vali1/tower/drop398944/best_model.pth'))
    # vali2
    model2 = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=116, dv=61,
        num_heads=2, dropout=0.589,
        act_name='leakyrelu'
    ).cuda()
    model2.load_state_dict(
        torch.load(parent_path + 'trained_models/vali2/tower/lgs_drop589-tsb1/second_model.pth'))
    # vali3
    model3 = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=55, dv=149,
        num_heads=6, dropout=0.242,
        act_name='leakyrelu'
    ).cuda()
    model3.load_state_dict(
        torch.load(parent_path + 'trained_models/vali3/tower/lgs_drop241/second_model.pth'))
    # vali4
    model4 = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=64, dv=105,
        num_heads=1, dropout=0.232,
        act_name='leakyrelu'
    ).cuda()
    model4.load_state_dict(
        torch.load(path + '/vali4/tower/drop232045/best_model.pth'))
    # vali5
    model5 = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=141, dv=96,
        num_heads=3, dropout=0.559,
        act_name='leakyrelu'
    ).cuda()
    model5.load_state_dict(
        torch.load(parent_path + 'trained_models/vali5/tower/lgs_drop559/best_model.pth'))
    # vali6
    model6 = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=101, dv=99,
        num_heads=1, dropout=0.272,
        act_name='leakyrelu'
    ).cuda()
    model6.load_state_dict(
        torch.load(path + 'vali6/tower/drop271766/best_model.pth'))

    bag_model = BaggingModel(
        [model1, model2, model3, model4, model5, model6],
        # [0.9897, 0.9719, 0.9676, 0.9837, 0.9752, 0.9777]
    )
    return bag_model
