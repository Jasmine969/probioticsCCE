import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy as dc
from torchmetrics.functional import r2_score
import time
import my_functions as mf
from tqdm import tqdm
import math


class SimpleTrainer:
    def __init__(self, net, batch_size, num_epoch, optimizer, criterion,
                 mode, flooding, seed=623,
                 device='cuda', tsb_track=None,
                 print_interval=5, early_stop=0):
        self.net = net
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = mode  # 'train' 'adjust'
        self.flooding = flooding
        self.early_stop = early_stop
        self.tsb_track = tsb_track
        if tsb_track is not None:
            self.writer = SummaryWriter(tsb_track)
        else:
            self.writer = None
        self.max_vali_acc = -2
        self.sync_train_acc = -2
        self.max_vali_epoch = 0
        self.best_net = dc(net)
        self.train_dataloader = None
        self.print_interval = print_interval
        self.device = device  # 'cpu' 'cuda'
        self.vali_acc_track = []
        mf.setup_seed(seed)

    def _train(self):
        for xb_train, tb_train in self.train_dataloader:
            predb_train = self.net(xb_train)
            loss_b = (self.criterion(predb_train, tb_train)
                      - self.flooding).abs() + self.flooding
            self.optimizer.zero_grad()
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=3, norm_type=float('inf'))
            self.optimizer.step()

    def _eval(self, x_train, x_vali, t_train, t_vali):
        with torch.no_grad():
            pred_train = self.net(x_train)
            pred_vali = self.net(x_vali)
            loss = (self.criterion(pred_train, t_train)
                    - self.flooding).abs() + self.flooding
            train_acc = r2_score(pred_train, t_train).item()
            vali_acc = r2_score(pred_vali, t_vali).item()
            return loss, train_acc, vali_acc

    def train(self, x_train, x_vali, t_train, t_vali):
        t_0 = time.time()
        if self.device == 'cuda':
            if x_train.device.type != 'cuda':
                x_train = x_train.cuda()
                x_vali = x_vali.cuda()
                t_train = t_train.cuda()
                t_vali = t_vali.cuda()
            self.net.cuda()
        self.train_dataloader = DataLoader(TensorDataset(x_train, t_train),
                                           batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epoch):
            self.net.train()
            self._train()
            self.net.eval()
            loss, train_acc, vali_acc = self._eval(x_train, x_vali, t_train, t_vali)
            if self.writer is not None:
                self.writer.add_scalar('loss/epoch', loss, epoch + 1)
                self.writer.add_scalars('accuracy', {'train': train_acc,
                                                     'vali': vali_acc}, epoch + 1)
            if self.mode == 'train':
                if not (epoch + 1) % self.print_interval:
                    print(f'epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.3f}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}, '
                          f'time_elapsed:{time.time() - t_0:4.1f}')
            else:
                self.vali_acc_track.append(vali_acc)
            update_cond = train_acc > self.sync_train_acc - 0.05 and vali_acc > self.max_vali_acc
            if update_cond:
                self.best_net = dc(self.net)
                self.max_vali_acc = vali_acc
                self.sync_train_acc = train_acc
                self.max_vali_epoch = epoch
                if self.mode == 'train':
                    print('-' * 50)
                    print(f'\033[0;33;40mHere comes a higher vali_acc, epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.3f}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}\033[0m')
                    print('-' * 50)
            if self.early_stop:
                if epoch - self.max_vali_epoch > self.early_stop:
                    break
        print('=' * 50)
        print(f'\033[0;32mThe highest vali_acc is {self.max_vali_acc:6.4f}'
              f' at {self.max_vali_epoch + 1} epoch!\033[0m')
        return self.best_net, self.net, self.max_vali_acc, self.vali_acc_track


class VaryLenInputTrainerLSTM(SimpleTrainer):
    """
    This trainer process input which is in the form of a list
     with tensors varying in their lengths.
     Also, it can only train RNN, LSTM and GRU.
    """

    def __init__(
            self, net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed=623,
            device='cuda', tsb_track=None,
            print_interval=5, early_stop=0,
            threshold=None, s_max=None, s_min=None,  # human intervention
            scalar=None, lg_ori=True, mix_acc=True
    ):
        super(VaryLenInputTrainerLSTM, self).__init__(
            net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed,
            device, tsb_track,
            print_interval, early_stop)
        # self.train_dataloader is the DataLoader of the training set during training
        self.eval_train_dataloader = None  # DataLoader of the training set during evaluation
        self.eval_vali_dataloader = None  # DataLoader of the validation set during evaluation
        self.threshold = threshold
        self.s_max = s_max
        self.s_min = s_min
        self.second_best_net = None
        self.inv_trans_dict = {
            'lg_ori': lg_ori,
            'scalar': scalar,
        }
        self.mix_acc = mix_acc

    def _itp_loss_calc(self, x_tb, length, calc_loss):
        xb, tb = mf.decompose_pack(x_tb, length)
        predb, _ = self.net(xb)
        predb = mf.my_pack(predb, length)
        loss_b = None
        if calc_loss:
            predb_cat = torch.cat(predb, dim=1)
            tb_cat = torch.cat(tb, dim=1)
            loss_b = self.criterion(predb_cat, tb_cat[:, :, [0]])
            loss_b[
                (~(tb_cat[:, :, [-1]].type(torch.bool))  # belong to interpolated labels
                 ) & (tb_cat[:, :, [1]] <= predb_cat
                      ) & (tb_cat[:, :, [2]] >= predb_cat)  # inside the feasible region
                ] = 0  # deemed reliable and loss=0
            loss_b = (loss_b.mean() - self.flooding).abs() - self.flooding
        return loss_b, predb, tb

    def _itp_loss_acc(self, x_tb, length, epoch, calc_loss):
        loss, pred, t = self._itp_loss_calc(x_tb, length, calc_loss)
        normal_acc, lg_acc = [], []
        for pred_each, t_each in zip(pred, t):
            pred_each = mf.inverse_transform(pred_each, **self.inv_trans_dict)
            t_each = dc(t_each)
            t_each[:, :, [0]] = mf.inverse_transform(t_each[:, :, [0]], **self.inv_trans_dict)
            real_ind = t_each[:, :, -1].type(torch.bool).squeeze()
            if self.threshold is not None and epoch > int(self.num_epoch * self.threshold):
                if epoch == int(self.num_epoch * self.threshold) + 1 and self.mode == 'train':
                    print('\033[0;34mIntervention begins!\033[0m')
                pred_each = mf.human_intervene(pred_each, self.s_max, real_ind, self.s_min)
            lg_acc.append(r2_score(pred_each[:, real_ind, :].squeeze(),
                                   t_each[:, real_ind, 0].squeeze()))
            normal_acc.append(r2_score(10 ** (pred_each[:, real_ind, :]).squeeze(),
                                       10 ** (t_each[:, real_ind, 0]).squeeze()))
        if self.mix_acc:
            total_acc = torch.Tensor(normal_acc + lg_acc)
        else:
            total_acc = torch.Tensor(
                lg_acc if self.inv_trans_dict['lg_ori'] else normal_acc)
        return loss, torch.mean(total_acc).item()

    def _optimization(self, loss_total):
        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        # count_grad = 0
        for each in self.net.parameters():
            # if each.grad is not None and each.grad.max().item() > 3:
            #     count_grad += 1
            #     print(name)
            # if count_grad > 2:
            #     raise ValueError('Gradient exploding')
            torch.nn.utils.clip_grad_norm_(
                each, max_norm=3, norm_type=float('inf'))
        self.optimizer.step()

    def _eval_var_len(self, epoch):
        with torch.no_grad():
            for x_tb_eval_train, length in self.eval_train_dataloader:
                loss, train_acc = self._itp_loss_acc(x_tb_eval_train, length, epoch, calc_loss=True)
            for x_tb_eval_vali, length in self.eval_vali_dataloader:
                _, vali_acc = self._itp_loss_acc(x_tb_eval_vali, length, epoch, calc_loss=False)
        return loss, train_acc, vali_acc

    def train_var_len(self, x_t_train, x_t_vali):
        t_0 = time.time()
        if self.device == 'cuda':
            if x_t_train[0].device.type != 'cuda':
                x_t_train = [x_each.cuda() for x_each in x_t_train]
                x_t_vali = [x_each.cuda() for x_each in x_t_vali]
            self.net.cuda()
        self.train_dataloader = DataLoader(
            mf.MyData(x_t_train),
            collate_fn=mf.collect_fn_no_added,
            batch_size=self.batch_size, shuffle=True)
        self.eval_train_dataloader = DataLoader(
            mf.MyData(x_t_train),
            collate_fn=mf.collect_fn_no_added,
            batch_size=len(x_t_train), shuffle=True)
        self.eval_vali_dataloader = DataLoader(
            mf.MyData(x_t_vali),
            collate_fn=mf.collect_fn_no_added,
            batch_size=len(x_t_vali), shuffle=True)

        for epoch in range(self.num_epoch) if \
                self.mode == 'train' else tqdm(range(self.num_epoch)):
            self.net.train()
            for x_tb_train, length in self.train_dataloader:
                loss_total, _, _ = self._itp_loss_calc(x_tb_train, length, calc_loss=True)
                self._optimization(loss_total)
            self.net.eval()
            loss, train_acc, vali_acc = self._eval_var_len(epoch)
            if self.writer is not None:
                self.writer.add_scalar('loss/epoch', loss, epoch + 1)
                self.writer.add_scalars('accuracy', {'train': train_acc,
                                                     'vali': vali_acc}, epoch + 1)
            if self.mode == 'train':
                if not (epoch + 1) % self.print_interval:
                    print(f'epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.4g}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}, '
                          f'time_elapsed:{time.time() - t_0:4.1f}')
            else:
                self.vali_acc_track.append(vali_acc)
            update_cond1 = train_acc > max(self.sync_train_acc - 0.02, vali_acc - 0.001
                                           ) and vali_acc > self.max_vali_acc and math.fabs(
                vali_acc - self.max_vali_acc) / (math.fabs(
                train_acc - self.sync_train_acc) + 1e-7) > 0.124
            update_cond2 = (train_acc > self.sync_train_acc + 0.02) and (
                    vali_acc > self.max_vali_acc - 0.01)  # train_acc rise a lot
            if update_cond1 or update_cond2:
                self.second_best_net = dc(self.best_net)
                self.best_net = dc(self.net)
                self.max_vali_acc = vali_acc
                self.sync_train_acc = train_acc
                self.max_vali_epoch = epoch
                if self.mode == 'train':
                    print('-' * 50)
                    print(f'\033[0;33;40mHere comes a higher vali_acc, epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.4g}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}\033[0m')
                    print('-' * 50)
            if self.early_stop:
                stop_cond1 = epoch - self.max_vali_epoch > self.early_stop
                stop_cond2 = (epoch / self.num_epoch > 0.4) and self.max_vali_acc < 0.2
                if stop_cond1 or stop_cond2:
                    break
        print('=' * 50)
        print(f'\033[0;32mThe highest vali_acc is {self.max_vali_acc:6.4f}'
              f' at {self.max_vali_epoch + 1} epoch!\033[0m')
        return self.best_net, self.net, self.max_vali_acc, self.vali_acc_track, \
               self.sync_train_acc, self.second_best_net


class VaryLenInputTrainerAttn(VaryLenInputTrainerLSTM):
    def __init__(
            self, net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed=623,
            itp_weight=1,  # weight of not real labels when calc loss
            last_weight=1,  # 1<=lw<=3,underline the prediction of initial points
            device='cuda', tsb_track=None,
            print_interval=5, early_stop=0,
            threshold=None, s_max=None, s_min=None,
            scalar=None, lg_ori=False, mix_acc=True
    ):
        super(VaryLenInputTrainerAttn, self).__init__(
            net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed,  # acc_mode: 'normal' or 'log' when calc. R2
            device, tsb_track, print_interval, early_stop,
            threshold=threshold, s_max=s_max, s_min=s_min,
            scalar=scalar, lg_ori=lg_ori, mix_acc=mix_acc
        )
        self.itp_weight = itp_weight
        self.last_weight = last_weight

    def _itp_loss_calc(self, x_tb, length, calc_loss):
        xb, tb = x_tb.split([3, 2], dim=-1)
        tb = mf.my_pack(tb, length)
        predb, _ = self.net(xb, length)
        predb = mf.my_pack(predb, length)
        loss_b = None
        if calc_loss:
            predb_cat = torch.cat(predb, dim=1)
            tb_cat = torch.cat(tb, dim=1)
            loss_b = self.criterion(predb_cat, tb_cat[:, :, [0]])
            real_tags = tb_cat[:, :, [-1]].type(torch.bool)
            loss_b[~real_tags] = loss_b[~real_tags] * self.itp_weight  # weaken the influences of itp labels
            loss_b = loss_b * torch.cat(
                [torch.linspace(1, self.last_weight, each_len) for each_len in length]
            ).cuda().reshape(1, -1, 1)  # linearly changing weights with len
            loss_b = (loss_b.mean() - self.flooding).abs() + self.flooding
        return loss_b, predb, tb


class TrainerAttn2Tower(VaryLenInputTrainerAttn):
    def __init__(
            self, net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed=623,
            itp_weight=1, last_weight=1,
            device='cuda', tsb_track=None,
            print_interval=5, early_stop=0,
            threshold=None
    ):
        super(TrainerAttn2Tower, self).__init__(
            net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed,
            itp_weight, last_weight,
            device, tsb_track,
            print_interval, early_stop, threshold
        )
        del self.last_weight
        del self.s_max, self.s_min, self.inv_trans_dict, self.mix_acc
        self.best_nm_train_acc = -2
        self.best_nm_vali_acc = -2
        self.best_lg_train_acc = -2
        self.best_lg_vali_acc = -2

    def _itp_loss_calc(self, x_tb, length, calc_loss):
        xb, tb = x_tb.split([3, 2], dim=-1)
        # tb1 = mf.my_pack(tb, length)  # normal + real tags
        # tb2 = [torch.log10(tbi[:, :, [0]]) for tbi in tb1]  # lg without real tags
        tb2 = mf.my_pack(tb, length)  # lg + real tags
        tb1 = [10 ** tbi[:, :, [0]] for tbi in tb2]  # normal without real tags
        predb, (predb1, predb2), _ = self.net(xb, length)  # predb == [out1, out2]
        predb1 = mf.my_pack(predb1, length)
        predb2 = mf.my_pack(predb2, length)
        loss_b = None
        if calc_loss:
            predb1_cat = torch.cat(predb1, dim=1)
            tb1_cat = torch.cat(tb1, dim=1)
            predb2_cat = torch.cat(predb2, dim=1)
            tb2_cat = torch.cat(tb2, dim=1)
            # loss_b = self.criterion(
            #     (predb1_cat, predb2_cat),
            #     (tb1_cat[:, :, [0]], tb2_cat[:, :, [0]])
            # )
            loss_b = self.criterion(predb1_cat, tb1_cat[:, :, [0]]
                                    ) + self.criterion(predb2_cat, tb2_cat[:, :, [0]])
            real_tags = tb2_cat[:, :, [-1]].type(torch.bool)
            loss_b[~real_tags] = loss_b[~real_tags] * self.itp_weight  # weaken the influences of itp labels
            loss_b = (loss_b.mean() - self.flooding).abs() + self.flooding
        return loss_b, predb, tb2  # predb:normal tb2:lg

    def _itp_loss_acc(self, x_tb, length, epoch, calc_loss):
        # pred (normal)
        loss, pred, t = self._itp_loss_calc(x_tb, length, calc_loss)
        pred = mf.my_pack(pred, length)
        normal_acc, lg_acc = [], []
        for pred_each, t_each in zip(pred, t):
            pred_each = mf.inverse_transform(
                pred_each,
                lg_ori=False
            )
            t_each = dc(t_each)
            t_each[:, :, [0]] = mf.inverse_transform(
                t_each[:, :, [0]],
                lg_ori=True
            )
            real_ind = t_each[:, :, 1].type(torch.bool).squeeze()
            if self.threshold is not None and epoch > int(self.num_epoch * self.threshold):
                if epoch == int(self.num_epoch * self.threshold) + 1 and self.mode == 'train':
                    print('\033[0;34mIntervention begins!\033[0m')
                pred_each = mf.human_intervene(pred_each, 1, real_ind, False, 1e-7)
            else:
                pred_each.clamp_min_(1e-7)
            normal_acc.append(r2_score(pred_each[:, real_ind, :].squeeze(),
                                       t_each[:, real_ind, 0].squeeze()))
            lg_acc.append(r2_score(torch.log10(pred_each[:, real_ind, :]).squeeze(),
                                   torch.log10(t_each[:, real_ind, 0]).squeeze()))
        normal_acc = torch.mean(torch.Tensor(normal_acc))
        lg_acc = torch.mean(torch.Tensor(lg_acc))
        return loss, normal_acc, lg_acc

    def _eval_var_len(self, epoch):
        with torch.no_grad():
            for x_tb_eval_train, length in self.eval_train_dataloader:
                loss, train_acc_normal, train_acc_lg = self._itp_loss_acc(
                    x_tb_eval_train, length, epoch, calc_loss=True)
            for x_tb_eval_vali, length in self.eval_vali_dataloader:
                _, vali_acc_normal, vali_acc_lg = self._itp_loss_acc(
                    x_tb_eval_vali, length, epoch, calc_loss=False)
        return loss, (train_acc_normal, vali_acc_normal), (train_acc_lg, vali_acc_lg)

    def train_var_len(self, x_t_train, x_t_vali):
        t_0 = time.time()
        if self.device == 'cuda':
            if x_t_train[0].device.type != 'cuda':
                x_t_train = [x_each.cuda() for x_each in x_t_train]
                x_t_vali = [x_each.cuda() for x_each in x_t_vali]
            self.net.cuda()
        self.train_dataloader = DataLoader(
            mf.MyData(x_t_train),
            collate_fn=mf.collect_fn_no_added,
            batch_size=self.batch_size, shuffle=True)
        self.eval_train_dataloader = DataLoader(
            mf.MyData(x_t_train),
            collate_fn=mf.collect_fn_no_added,
            batch_size=len(x_t_train), shuffle=False)
        self.eval_vali_dataloader = DataLoader(
            mf.MyData(x_t_vali),
            collate_fn=mf.collect_fn_no_added,
            batch_size=len(x_t_vali), shuffle=False)

        for epoch in range(self.num_epoch) if \
                self.mode == 'train' else tqdm(range(self.num_epoch)):
            self.net.train()
            for x_tb_train, length in self.train_dataloader:
                loss_total, _, _ = self._itp_loss_calc(x_tb_train, length, calc_loss=True)
                self._optimization(loss_total)
            self.net.eval()
            loss, (train_acc_normal, vali_acc_normal
                   ), (train_acc_lg, vali_acc_lg) = self._eval_var_len(epoch)
            if self.writer is not None:
                self.writer.add_scalar('loss/epoch', loss, epoch + 1)
                self.writer.add_scalars(
                    'accuracy_normal', {'train': train_acc_normal,
                                        'vali': vali_acc_normal}, epoch + 1)
                self.writer.add_scalars(
                    'accuracy_lg', {'train': train_acc_lg,
                                    'vali': vali_acc_lg}, epoch + 1)
            if self.mode == 'train':
                if not (epoch + 1) % self.print_interval:
                    print(f'epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.4g}, '
                          f'train_acc:{train_acc_normal:6.4f}, {train_acc_lg:6.4f}, '
                          f'vali_acc:{vali_acc_normal:6.4f}, {vali_acc_lg:6.4f}, '
                          f'time_elapsed:{time.time() - t_0:4.1f}')
            else:
                self.vali_acc_track.append((vali_acc_normal, vali_acc_lg))
            train_acc = 0.4 * train_acc_normal + 0.6 * train_acc_lg
            vali_acc = 0.4 * vali_acc_normal + 0.6 * vali_acc_lg
            update_cond1 = train_acc > max(self.sync_train_acc - 0.02, vali_acc - 0.001
                                           ) and vali_acc > self.max_vali_acc and math.fabs(
                vali_acc - self.max_vali_acc) / (math.fabs(
                train_acc - self.sync_train_acc) + 1e-7) > 0.124
            update_cond2 = (train_acc > self.sync_train_acc + 0.02) and (
                    vali_acc > self.max_vali_acc - 0.01)  # train_acc rise a lot
            if update_cond1 or update_cond2:
                self.second_best_net = dc(self.best_net)
                self.best_net = dc(self.net)
                self.max_vali_acc = vali_acc
                self.sync_train_acc = train_acc
                self.max_vali_epoch = epoch
                self.best_nm_train_acc = train_acc_normal
                self.best_nm_vali_acc = vali_acc_normal
                self.best_lg_train_acc = train_acc_lg
                self.best_lg_vali_acc = vali_acc_lg
                if self.mode == 'train':
                    print('-' * 50)
                    print(f'\033[0;33;40mHere comes a higher vali_acc, epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.4g}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}\033[0m')
                    print('-' * 50)
            if self.early_stop:
                stop_cond1 = epoch - self.max_vali_epoch > self.early_stop
                stop_cond2 = (epoch / self.num_epoch > 0.4) and self.max_vali_acc < 0.2
                if stop_cond1 or stop_cond2:
                    break
        print('=' * 50)
        print(f'\033[0;32mThe highest vali_acc is {self.max_vali_acc:6.4f}'
              f' at {self.max_vali_epoch + 1} epoch!\033[0m')
        return {
            'best_net': self.best_net, 'net': self.net,
            'max_vali_acc': self.max_vali_acc,
            'sync_train_acc': self.sync_train_acc,
            'second_best_net': self.second_best_net,
            'best_nm_train_acc': self.best_nm_train_acc,
            'best_nm_vali_acc': self.best_nm_vali_acc,
            'best_lg_train_acc': self.best_lg_train_acc,
            'best_lg_vali_acc': self.best_lg_vali_acc,
            'max_vali_epoch': self.max_vali_epoch + 1
        }


class VaryLenInputTrainerAttnAdded(VaryLenInputTrainerAttn):
    def __init__(
            self, net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed=623, real_weight=1,  # weight of real label when calc loss
            device='cuda', tsb_track=None,
            print_interval=5, early_stop=0,
            threshold=None, s_max=None, s_min=None
    ):
        super(VaryLenInputTrainerAttn, self).__init__(
            net, batch_size, num_epoch, optimizer, criterion,
            mode, flooding, seed,  # acc_mode: 'normal' or 'log' when calc. R2
            device, tsb_track, print_interval, early_stop,
            threshold=threshold, s_max=s_max, s_min=s_min
        )
        self.real_weight = real_weight
        self.second_best_net = None

    def _pred(self, x_tb, length):
        xb, tb = x_tb.split([3, 2], dim=-1)
        tb = mf.my_pack(tb, length, out_last=True)
        predb = self.net(xb, length, self.mode)
        predb = mf.my_pack(predb, length, out_last=True)
        return predb, tb

    def _itp_loss_calc_added(self, predb, tb):
        predb_cat = torch.cat(predb, dim=1)
        tb_cat = torch.cat(tb, dim=1)
        loss_b = (self.criterion(predb_cat, tb_cat[:, :, [0]])
                  - self.flooding).abs() + self.flooding
        real_tags = tb_cat[:, :, [-1]].type(torch.bool)
        loss_b[real_tags] = loss_b[real_tags] * self.real_weight  # underline the prediction of real labels
        loss_b = loss_b.mean()
        return loss_b

    def _itp_loss_acc_added(self, pred, t, epoch, ind, lengths):
        acc = []
        pred, t = mf.sort_split_tv(pred, t, ind, lengths)
        for pred_each, t_each in zip(pred, t):
            if self.threshold is not None and epoch > int(self.num_epoch * self.threshold):
                if epoch == int(self.num_epoch * self.threshold) + 1 and self.mode == 'train':
                    print('\033[0;34mIntervention begins!\033[0m')
                pred_each = mf.human_intervene(pred_each, self.s_max, self.s_min)
            real_ind = t_each[:, :, -1].type(torch.bool).squeeze()
            acc.append(r2_score(pred_each[:, real_ind, :].squeeze(),
                                t_each[:, real_ind, 0].squeeze()))
        return torch.mean(torch.Tensor(acc)).item()

    def _eval_var_len(self, epoch):
        with torch.no_grad():
            # train_acc
            pred, t, indices, lengths = [], [], [], []
            for ind, x_tb_eval_train, length in self.eval_train_dataloader:
                predb, tb = self._pred(x_tb_eval_train, length)
                pred.extend(predb)
                t.extend(tb)
                indices.extend(ind)
                lengths.extend(length)
            loss = self._itp_loss_calc_added(pred, t)
            train_acc = self._itp_loss_acc_added(pred, t, epoch, indices, lengths)
            #  vali_acc
            pred, t, indices = [], [], []
            for ind, x_tb_eval_vali, length in self.eval_vali_dataloader:
                predb, tb = self._pred(x_tb_eval_vali, length)
                pred.extend(predb)
                t.extend(tb)
                indices.extend(ind)
            vali_acc = self._itp_loss_acc_added(pred, t, epoch, indices, lengths)
        return loss, train_acc, vali_acc

    def train_var_len(self, x_t_train, x_t_vali):
        t_0 = time.time()
        x_t_train_ind, x_t_train = x_t_train
        x_t_vali_ind, x_t_vali = x_t_vali
        if self.device == 'cuda':
            if x_t_train[0].device.type != 'cuda':
                x_t_train = [x_each.cuda() for x_each in x_t_train]
                x_t_vali = [x_each.cuda() for x_each in x_t_vali]
            self.net.cuda()
        self.train_dataloader = DataLoader(
            mf.MyData(x_t_train, x_t_train_ind),
            collate_fn=mf.collect_fn_added,
            batch_size=self.batch_size, shuffle=True)
        self.eval_train_dataloader = DataLoader(
            mf.MyData(x_t_train, x_t_train_ind),
            collate_fn=mf.collect_fn_added,
            batch_size=400, shuffle=False)
        self.eval_vali_dataloader = DataLoader(
            mf.MyData(x_t_vali, x_t_vali_ind),
            collate_fn=mf.collect_fn_added,
            batch_size=400, shuffle=False)

        for epoch in range(self.num_epoch) if \
                self.mode == 'train' else tqdm(range(self.num_epoch)):
            self.net.train()
            for ind, x_tb_train, length in self.train_dataloader:
                loss_total, _, _ = self._itp_loss_calc(x_tb_train, length, calc_loss=True)
                self._optimization(loss_total)
            self.net.eval()
            loss, train_acc, vali_acc = self._eval_var_len(epoch)
            if self.writer is not None:
                self.writer.add_scalar('loss/epoch', loss, epoch + 1)
                self.writer.add_scalars('accuracy', {'train': train_acc,
                                                     'vali': vali_acc}, epoch + 1)
            if self.mode == 'train':
                if not (epoch + 1) % self.print_interval:
                    print(f'epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.3f}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}, '
                          f'time_elapsed:{time.time() - t_0:4.1f}')
            else:
                self.vali_acc_track.append(vali_acc)
            update_cond1 = train_acc > max(self.sync_train_acc - 0.03, vali_acc - 0.02
                                           ) and vali_acc > self.max_vali_acc
            update_cond2 = (train_acc > self.sync_train_acc + 0.02) and (
                    vali_acc > self.max_vali_acc - 0.01)
            if update_cond1 or update_cond2:
                self.second_best_net = dc(self.best_net)
                self.best_net = dc(self.net)
                self.max_vali_acc = vali_acc
                self.sync_train_acc = train_acc
                self.max_vali_epoch = epoch
                if self.mode == 'train':
                    print('-' * 50)
                    print(f'\033[0;33;40mHere comes a higher vali_acc, epoch:{epoch + 1}/{self.num_epoch}, '
                          f'loss:{loss:6.3f}, '
                          f'train_acc:{train_acc:6.4f}, '
                          f'vali_acc:{vali_acc:6.4f}\033[0m')
                    print('-' * 50)
            if self.early_stop:
                if epoch - self.max_vali_epoch > self.early_stop:
                    break
        print('=' * 50)
        print(f'\033[0;32mThe highest vali_acc is {self.max_vali_acc:6.4f}'
              f' at {self.max_vali_epoch + 1} epoch!\033[0m')
        return self.best_net, self.net, self.max_vali_acc, \
               self.vali_acc_track, self.sync_train_acc, self.second_best_net
