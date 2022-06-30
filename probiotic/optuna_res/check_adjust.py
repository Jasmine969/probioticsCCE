import torch
import pickle
from torch import nn, optim
from trainers import TrainerAttn2Tower
from probiotic.MyNN import ConvAttn2Tower
import my_functions as mf
from torch.utils.data import DataLoader

with open('../../probiotic/pickle_data/test35_lgs_non-tilde.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    # use group 1 as the validation set
    x_t_train, x_t_vali = mf.split_train_vali(dct['ft_s_tv'], 1)
    s_scalar = dct['s_scalar']
model = ConvAttn2Tower(
    in_channels=3, kernel_size=3,
    dk=146, dv=111, num_heads=1, dropout=0.402,
    act_name='leakyrelu'
).cuda()
model_path = '../../probiotic/optuna_res/test35/vali1/tower/drop401682'
model.load_state_dict(torch.load(model_path + '/best_model.pth'))
model.eval()
trainer = TrainerAttn2Tower(
    net=model, batch_size=2, num_epoch=500,
    optimizer=optim.Adam(model.parameters(), lr=0.01),
    criterion=nn.MSELoss(), mode='train', flooding=0.001,
    threshold=-1
)
length = [270, 270, 300, 120, 400, 400]
x_t_train = [x_each.cuda() for x_each in x_t_train]
x_t_vali = [x_each.cuda() for x_each in x_t_vali]
eval_train_dataloader = DataLoader(
    mf.MyData(x_t_train),
    collate_fn=mf.collect_fn_no_added,
    batch_size=len(x_t_train), shuffle=True)
eval_vali_dataloader = DataLoader(
    mf.MyData(x_t_vali),
    collate_fn=mf.collect_fn_no_added,
    batch_size=len(x_t_vali), shuffle=True)
with torch.no_grad():
    for x_tb_eval_train, length in eval_train_dataloader:
        loss, train_acc_normal, train_acc_lg = trainer._itp_loss_acc(
            x_tb_eval_train, length=length,
            epoch=0, calc_loss=False
        )
    for x_tb_eval_vali, length in eval_vali_dataloader:
        _, vali_acc_normal, vali_acc_lg = trainer._itp_loss_acc(
            x_tb_eval_vali, length=length,
            epoch=0, calc_loss=False)
normal_acc = 0.65 * vali_acc_normal + 0.35 * train_acc_normal
lg_acc = 0.65 * vali_acc_lg + 0.35 * train_acc_lg
print(normal_acc, lg_acc)