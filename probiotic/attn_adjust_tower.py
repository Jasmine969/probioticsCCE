import pickle
from MyNN import ConvAttn2Tower
from trainers import TrainerAttn2Tower
from torch import nn, optim
import optuna
from plotly import offline
import pandas as pd
import torch
import my_functions as mf
import os
import gc

with open('pickle_data/test35_lgs_non-tilde.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    # use group 2 as the validation set
    ft_s_train, ft_s_vali = mf.split_train_vali(dct['ft_s_tv'], 2)
    s_scalar = dct['s_scalar']

epoch_num = 1000
path = 'optuna_res/test35/vali2/tower/'
if not os.path.exists(path):
    os.makedirs(path)


def objective(trial):
    batch_size = trial.suggest_int('bs', 1, 2)
    # batch_size = 1
    lr = trial.suggest_float('lr', 0.0002, 0.01, log=True)
    itv = trial.suggest_categorical('itv', ['itv', 'none'])
    # itv = 'itv'
    threshold = None
    if itv == 'itv':
        threshold = 1 / trial.suggest_float('threshold', 2, 10, log=True)
    weight_decay = trial.suggest_float('decay', 1e-6, 4e-3, log=True)
    flooding = trial.suggest_float('flood', 0.001, 0.05, log=True)
    itp_weight = trial.suggest_float('itp_weight', 0.1, 1)
    # itp_weight = 1
    drop = trial.suggest_float('drop', 0.1, 0.7)
    pgd_epsilon = trial.suggest_float('epsilon', 0.001, 1., log=True)
    pgd_alpha = trial.suggest_float('alpha', 0.0003, 0.3, log=True)
    model = ConvAttn2Tower(
        in_channels=3, kernel_size=3,
        dk=trial.suggest_int('dk', 30, 150),
        dv=trial.suggest_int('dv', 30, 150),
        num_heads=trial.suggest_int('head', 1, 6),
        dropout=drop,
        act_name='leakyrelu'
    )
    criterion = nn.MSELoss(reduction='none')
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = optim.Adam(
        [{'params': weight_p, 'weight_decay': weight_decay},
         {'params': bias_p, 'weight_decay': 0},
         ], lr=lr
    )
    model_path = path + 'drop' + str(drop)[2:8]
    tsb_path = model_path + '/tsb_res'
    os.makedirs(model_path)
    os.makedirs(tsb_path)
    trainer = TrainerAttn2Tower(
        net=model, batch_size=batch_size,
        num_epoch=epoch_num, optimizer=optimizer, print_interval=1,
        criterion=criterion, mode='adjust', seed=7,
        flooding=flooding, device='cuda', itp_weight=itp_weight,
        threshold=threshold, tsb_track=tsb_path,
        early_stop=20000, pgd_epsilon=pgd_epsilon, pgd_alpha=pgd_alpha
    )
    dct_metrics = trainer.train_var_len(
        x_t_train=ft_s_train, x_t_vali=ft_s_vali)
    normal_acc = dct_metrics['best_nm_vali_acc'
                 ] * 0.65 + dct_metrics['best_nm_train_acc'] * 0.35
    lg_acc = dct_metrics['best_lg_vali_acc'
             ] * 0.65 + dct_metrics['best_lg_train_acc'] * 0.35
    if dct_metrics['best_net'] is not None:
        torch.save(dct_metrics['best_net'].state_dict(), model_path + '/best_model.pth')
    if dct_metrics['net'] is not None:
        torch.save(dct_metrics['net'].state_dict(), model_path + '/last_model.pth')
    if dct_metrics['second_best_net'] is not None:
        torch.save(dct_metrics['second_best_net'].state_dict(), model_path + '/second_model.pth')
    del model, dct_metrics, trainer, model_path, tsb_path, optimizer, criterion
    gc.collect()
    return normal_acc, lg_acc


study = optuna.create_study(
    study_name='attn_prob',
    storage='sqlite:///' + path + 'lgs-pgd.sqlite3',
    directions=['maximize', 'maximize']
)
study.optimize(objective, n_trials=50)
pareto = optuna.visualization.plot_pareto_front(
    study, target_names=['normal_acc', 'lg_acc'])
offline.plot(pareto)
writer = pd.ExcelWriter(path + 'lgs_pgd.xlsx')
study.trials_dataframe().to_excel(writer, index=None)
writer.save()
