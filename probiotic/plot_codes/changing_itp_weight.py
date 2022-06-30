import pickle
from MyNN import ConvAttn2Tower
from probioticsCCE.trainers import TrainerAttn2Tower
from torch import nn, optim
from probioticsCCE import my_functions as mf
import numpy as np

# test group: 3 5
with open('pickle_data/test35_lgs_non-tilde.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    # use group 4 as the validation set
    ft_s_train, ft_s_vali = mf.split_train_vali(dct['ft_s_tv'], 4)
    s_scalar = dct['s_scalar']


def objective(itp_weight):
    hypers = {
        'batch_size': 2,
        'epoch_num': 2000,
        'learning_rate': 7.85e-4,
        'weight_decay': 3.25e-5,
        'flooding': 0.00203,
        'threshold': 1 / 5.7,
        'itp_weight': itp_weight,
        'dk': 64,
        'dv': 105,
        'num_heads': 1,
        'dropout': 0.232,
        'act_name': 'leakyrelu',
    }
    model = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=hypers['dk'], dv=hypers['dv'],
        num_heads=hypers['num_heads'],
        dropout=hypers['dropout'],
        act_name=hypers['act_name']
    )
    mf.params_num(model)
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = optim.Adam(
        [{'params': weight_p, 'weight_decay': hypers['weight_decay']},
         {'params': bias_p, 'weight_decay': 0}
         ], lr=hypers['learning_rate']
    )
    criterion = nn.MSELoss(reduction='none')
    trainer = TrainerAttn2Tower(
        net=model, batch_size=hypers['batch_size'],
        num_epoch=hypers['epoch_num'], optimizer=optimizer,
        criterion=criterion, mode='train', seed=7,
        flooding=hypers['flooding'], device='cuda',
        itp_weight=hypers['itp_weight'],
        tsb_track=None,
        print_interval=5, early_stop=1000,
        threshold=hypers['threshold'],
    )
    dct_metrics = trainer.train_var_len(
        x_t_train=ft_s_train, x_t_vali=ft_s_vali)

    return dct_metrics['best_nm_train_acc'], dct_metrics[
        'best_nm_vali_acc'], dct_metrics[
        'best_lg_train_acc'], dct_metrics[
        'best_lg_vali_acc']


itp_weights = np.linspace(0, 1, 21)
nm_trains, nm_valis = [], []
lg_trains, lg_valis = [], []
for itp_w in itp_weights:
    print(f'itp_weight={itp_w}' + '~' * 10)
    nm_train, nm_vali, lg_train, lg_vali = objective(itp_w)
    nm_trains.append(nm_train)
    nm_valis.append(nm_vali)
    lg_trains.append(lg_train)
    lg_valis.append(lg_vali)
with open('pickle_data/change_itp_weight.pkl', 'wb') as pf2:
    pickle.dump({'nm_trains': nm_trains, 'nm_valis': nm_valis,
                 'lg_trains': lg_trains, 'lg_valis': lg_valis}, pf2)
