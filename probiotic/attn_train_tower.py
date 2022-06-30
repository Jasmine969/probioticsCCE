import pickle
import torch
from MyNN import ConvAttn2Tower
from probioticsCCE.trainers import TrainerAttn2Tower
from torch import optim, nn
from probioticsCCE import my_functions as mf
import os

# test group: 3 5
with open('pickle_data/test35_lgs_non-tilde.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    # use group 2 as the validation set
    ft_s_train, ft_s_vali = mf.split_train_vali(dct['ft_s_tv'], 2)
    s_scalar = dct['s_scalar']
path = 'trained_models/vali2/tower/lgs_pgd'
tsb_path = path + '/tensorboard_res'
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(tsb_path)
# tsb_path = None
hypers = {
    'batch_size': 2,
    'epoch_num': 1000,
    'learning_rate': 2.79e-3,
    'weight_decay': 2.96e-5,
    'flooding': 6.41e-3,
    'threshold': 1 / 5.1,
    'itp_weight': 0.973,
    'dk': 116,
    'dv': 61,
    'num_heads': 2,
    'dropout': 0.589,
    'act_name': 'leakyrelu',
    'pgd_epsilon': 1.,
    'pgd_alpha': .3
}

model = ConvAttn2Tower(
    in_channels=3, kernel_size=3,
    dk=hypers['dk'], dv=hypers['dv'],
    num_heads=hypers['num_heads'],
    dropout=hypers['dropout'],
    act_name=hypers['act_name']
)
mf.params_num(model)
criterion = nn.MSELoss(reduction='none')
weight_p, bias_p = [], []
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
optimizer = optim.Adam(
    [{'params': weight_p, 'weight_decay': hypers['weight_decay']},
     {'params': bias_p, 'weight_decay': 0},
     ], lr=hypers['learning_rate']
)
trainer = TrainerAttn2Tower(
    net=model, batch_size=hypers['batch_size'],
    num_epoch=hypers['epoch_num'], optimizer=optimizer,
    criterion=criterion, mode='train', seed=7,
    flooding=hypers['flooding'], device='cuda',
    itp_weight=hypers['itp_weight'],
    tsb_track=tsb_path,
    print_interval=5, early_stop=1000,
    threshold=hypers['threshold'],
    pgd_epsilon=hypers['pgd_epsilon'],
    pgd_alpha=hypers['pgd_alpha']
)
dct_metrics = trainer.train_var_len(
    x_t_train=ft_s_train, x_t_vali=ft_s_vali)
if dct_metrics is not None:
    torch.save(dct_metrics['best_net'].state_dict(), path + '/best_model.pth')
    torch.save(dct_metrics['second_best_net'].state_dict(), path + '/second_model.pth')

    with open(path + '/hyper_record.txt', 'w') as f:
        for key, val in hypers.items():
            f.write(key + ': ' + str(val) + '\n')
        f.write('\n')
        f.write(f'max_vali_acc: {dct_metrics["max_vali_acc"]:.4f},'
                f' train_sync_acc: {dct_metrics["sync_train_acc"]:.4f}'
                f' max_vali_epoch: {dct_metrics["max_vali_epoch"]}')
