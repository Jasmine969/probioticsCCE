import pickle
from MyNN import MyLSTM
from probioticsCCE.trainers import VaryLenInputTrainerLSTM
from torch import nn, optim
import math
import probioticsCCE.my_functions as mf

with open('pickle_data\\zscore_train_val_test_scalar.pkl', 'rb') as pf1:
    dct = pickle.load(pf1)
    # use group 1 as the validation set
    ft_s_train = dct['ft_s_tv'][1:]
    ft_s_val = [dct['ft_s_tv'][0]]
    s_scalar = dct['s_scalar']
max_t = ((0 - s_scalar.mean_)/s_scalar.var_**0.5)[0]
batch_size, epoch_num = 1, 500
threshold = int(epoch_num / 3 * math.ceil(len(ft_s_train) / batch_size))
learning_rate = 0.00133
weight_decay, flooding = 1.208e-5, 2.127e-5

model = MyLSTM(input_dim=4, lstm_dim=50,
               fc_dim=50, lstm_layer=2, drop=0.160,max_t=max_t, threshold=threshold).cuda()
mf.params_num(model)
weight_p, bias_p = [], []
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
optimizer = optim.Adam(
    [{'params': weight_p, 'weight_decay': weight_decay},
     {'params': bias_p, 'weight_decay': 0}
     ], lr=learning_rate
)
criterion = nn.MSELoss(reduction='none')
trainer = VaryLenInputTrainerLSTM(
    net=model, batch_size=batch_size,
    num_epoch=epoch_num, optimizer=optimizer,
    criterion=criterion, mode='train', seed=7,
    flooding=flooding, device='cuda', acc_mode='normal',
    tsb_track=None, print_interval=1, early_stop=200
)
trainer.train_var_len(x_t_train=ft_s_train, x_t_vali=ft_s_val)
