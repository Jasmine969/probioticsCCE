import math
from torch import nn
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from torch.nn.utils import rnn
from matplotlib import font_manager as fm, pyplot as plt
from sklearn.metrics import r2_score
from copy import deepcopy as dc


def collect_fn_added(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    ind, data = list(zip(*data))
    ind, data = list(ind), list(data)
    data_len = [len(each) for each in data]
    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return ind, data, data_len


def collect_fn_no_added(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_len = [len(each) for each in data]
    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_len


class MyData(Dataset):
    def __init__(self, train_x, ind=None):
        self.ind = ind
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        if self.ind is None:
            return self.train_x[item]
        else:
            return self.ind[item], self.train_x[item]


def my_pack(t, length, out_last=False):
    t = t.split(1, dim=0)  # split t into different groups
    if out_last:
        t = [t[idx][:, [val - 1], :] for idx, val in enumerate(length)]
    else:
        t = [t[idx][:, :val, :] for idx, val in enumerate(length)]  # pack t
    return t


def decompose_pack(x_t, length):
    """
    x means input, t means labels
    decompose [x,t] from DataLoader into x and t
    and pack t and x so that it contains no padding values
    :param length: list which stores lengths of t of different groups
    :param x_t: Tensor [x,t], 8 columns, 4 for x and 4 for t
    :return: x and t where x and t are packed respectively
    """
    x, t = x_t.split([4, 4], dim=-1)  # split x and t
    t = my_pack(t, length)
    x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)  # pack t
    return x, t


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def params_num(net):
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fK" % (total / 1e3))


def mono_decrease(x):
    m = np.tile(x, (x.size, 1)) + np.triu(np.ones(x.size) * 10, 1)
    out = m.min(axis=-1)
    return out


def human_intervene(s, s_max, real_ind, clip_increase, s_min=None):
    """
    s_0=s_max;clamp in [0,1] or [f(0),f(1)];mono decrease
    Note that logits and clamp can't be carried out simultaneously
    :param clip_increase: if True, clip the increase part and then mono decrease the start
    :param real_ind: indices of real labels
    :param s_min: normal -> 0, lg -> -inf, normal_scaled -> scale(0), lg_scaled -> -inf
    :param s_max: normal -> 1, lg -> 0, normal_scaled -> scale(1), lg_scaled -> scale(0)
    :param s: (b,seq,1)
    :return: (b,seq,1)
    """
    # set the initial value as s_max
    s[:, 0, :] = s_max
    real_ind_2 = torch.where(real_ind)[0][1].item()
    tmp_min, tmp_max = sorted(s[:, real_ind, :].squeeze().tolist()[:2])
    s[:, 0:real_ind_2 + 1, :] = torch.linspace(
        tmp_max, tmp_min, real_ind_2 + 1).reshape(1, -1, 1)
    # clamp
    if s_min is None:
        torch.clamp_max_(s, s_max)
    else:
        s = torch.clamp(s, s_min, s_max)
    if clip_increase:
        increase_ind = torch.zeros_like(s).type(torch.BoolTensor)
        increase_ind[:, 110:, :] = s[:, 110:, :] > s[:, 109:-1, :]
        if torch.sum(increase_ind):
            min_increase_ind = torch.where(increase_ind)[1][0].item()
            increase_ind = increase_ind[:, min_increase_ind:, :]
            # if torch.sum(increase_ind) == increase_ind.size(1):
            if torch.sum(increase_ind) / torch.numel(increase_ind) > 0.9:
                s = s[:, :min_increase_ind, :]
    # mono decrease
    b, seq, _ = s.size()
    m = s.repeat((1, 1, seq)) + torch.tril(
        torch.ones(b, seq, seq) * 1000, diagonal=-1).cuda()
    out = m.min(dim=1).values.unsqueeze(-1)
    return out


def sort_split_tv(pred: list, t: list, ind: list, length: list):
    total = list(zip(pred, t, ind, length))
    total.sort(key=lambda x: x[-1])
    pred_g, t_g = [], []
    for i in set(ind):
        pred, t, ind, length = list(zip(*list(
            filter(lambda x: x[-2] == i, total))))
        pred_g.append(torch.cat(pred, dim=1))
        t_g.append(torch.cat(t, dim=1))
    return pred_g, t_g


def sort_split_test(ind_pred_t):
    indices, pred_t = ind_pred_t
    start_ind = [indices.index(i) for i in set(indices)] + [len(indices)]
    # split pred and t
    tmp_len = [each.size(0) for each in pred_t]
    pred_total, t_total, real_ind_total = torch.cat(pred_t, dim=0).split([3, 1, 1], dim=1)
    pred_total = pred_total.split(tmp_len, dim=0)
    t_total = t_total.split(tmp_len, dim=0)
    real_ind_total = real_ind_total.type(torch.bool).split(tmp_len, dim=0)
    # split each group
    pred_g, t_g, real_ind_g, length_g = dict(), dict(), dict(), dict()
    for i, ind in enumerate(set(indices)):
        pred_g[str(ind)] = list(pred_total[start_ind[i]:start_ind[i + 1]])
        t_g[str(ind)] = list(t_total[start_ind[i]:start_ind[i + 1]])
        real_ind_g[str(ind)] = list(real_ind_total[start_ind[i]:start_ind[i + 1]])
        length_g[str(ind)] = list(tmp_len[start_ind[i]:start_ind[i + 1]])
    return pred_g, t_g, real_ind_g, length_g


def logit(x):
    x[x == 1] = 1 - 1e-4
    x = np.log(x / (1 - x))
    return x


def my_sigmoid(x):
    x = torch.sigmoid(x)
    x[x >= 1 - 1e-4] = 1.0
    return x


def my_sigmoid_np(x):
    x = 1 / (1 + np.exp(-x))
    x[x >= 1 - 1e-4] = 1.0
    return x


def put_first(lst: list, ind: int):
    """
    put an element at a certain index to the first place
    in order to implement my_diff and shuffle
    """
    new_indices = list(range(len(lst)))
    new_indices.pop(ind)
    new_indices = [ind] + new_indices
    new_lst = np.asarray(lst)[new_indices].tolist()
    recover_indices = list(range(1, len(lst)))
    recover_indices.insert(ind, 0)
    return new_indices, new_lst, recover_indices


# ================================diff
def my_torch_diff(x, dim=0):
    x = x.numpy()
    d = -np.diff(x, axis=dim)
    d = np.concatenate(
        (np.zeros(tuple(map(
            lambda x, y: x if x == y else int(math.fabs(x - y)), x.shape, d.shape
        ))), d), axis=dim
    )  # map func input(5,3,4)(5,2,4),output(5,1,4) to recover the shape
    d = torch.from_numpy(d).type(torch.FloatTensor)
    return d


# ================================ shuffle
def torch_shuffle_1d(x):
    idx = torch.randperm(x.numel())
    out = x[idx]
    return out


def torch_shuffle_2d0(x):
    """
    if the shape is (a,b), shuffle the 0th dim with the length of a
    :param x: 2d tensor
    :return: 2d tensor
    """
    assert x.ndim == 2
    idx = torch.randperm(x.size(0))
    out = x[idx, :]
    return out


def torch_shuffle(x, dim=None):
    nd = x.ndim
    if nd == 1:
        out = torch_shuffle_1d(x)
        return out
    if dim is None:
        out = torch_shuffle_1d(x.flatten()).reshape(x.size())
        return out
    else:
        shape = x.size()  # (a,b,c,d)
        new_ind, new_shape, recover_ind = put_first(list(shape), dim)
        out = x.permute(new_ind)  # (b,a,c,d) if dim=1
        out = out.reshape(shape[dim], -1)  # (b,a*c*d)
        out = torch_shuffle_2d0(out)  # (b,a*c*d)
        out = out.reshape(new_shape)  # (b,a,c,d)
        out = out.permute(recover_ind)  # (a,b,c,d)
        return out


def inverse_transform(y, lg_ori, scalar=None):
    if scalar is not None:
        y = torch.from_numpy(
            scalar.inverse_transform(y.cpu().squeeze(-1))
        ).type(torch.FloatTensor).unsqueeze(-1).cuda()
    if lg_ori:
        y = 10 ** y
    return y


class LgNormalLoss(nn.Module):
    def __init__(self, reduction):
        super(LgNormalLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.cp = math.log10(1 / math.log(10))  # critical point,-0.362

    def forward(self, y, t):
        cond = (y < 0) & (y > self.cp) & (t < 0) & (t > self.cp)
        y[cond] = 10 ** y[cond]
        t[cond] = 10 ** t[cond]
        loss = self.criterion(y, t)
        return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, reduction, w0: list = None):
        super(MultiTaskLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        # if w0 is None:
        #     self.log_vars = nn.Parameter(torch.zeros(2, ))
        # else:
        #     self.log_vars = torch.Tensor(w0)
        #     self.log_vars = nn.Parameter(torch.Tensor(w0))

    def forward(self, y, t):
        # loss0 = torch.exp(-self.log_vars[0]) * self.criterion(
        #     y[0], t[0]) + self.log_vars[0]
        # loss1 = torch.exp(-self.log_vars[1]) * self.criterion(
        #     y[1], t[1]) + self.log_vars[1]
        loss0 = self.criterion(y[0], t[0])
        # loss0 = 0
        loss1 = self.criterion(y[1], t[1])

        return loss0 + loss1


def split_train_vali(ft_s_tv, vali_code):
    assert 1 <= vali_code <= 6
    ft_s_vali = [ft_s_tv[vali_code - 1]]
    ft_s_train = ft_s_tv[:vali_code - 1] + ft_s_tv[vali_code:]
    return ft_s_train, ft_s_vali


def put_last(lst, group_code):
    tmp = lst.pop(group_code - 1)
    lst.append(tmp)
    return lst


def gen_hist_xticklabel(vali_code):
    group_class = ['(train)'] * 5 + ['(vali)']
    group_codes = put_last(list(range(1, 7)), vali_code)
    labels = [str(group_codes[i]) + ' ' + group_class[i] for i in range(6)]
    return labels


def pack_test(ft_s):
    ft = [ft_s[:, :3].type(torch.float)]
    s = ft_s[:, [3]].numpy()
    real_ind = ft_s[:, -1].numpy().astype(bool)
    length = [len(ft_s)]
    return ft, s, real_ind, length


def conv_smooth(x, win_size):
    assert win_size % 2 == 1  # must be an odd number
    x = x.astype(np.float)
    win = np.ones(win_size) / win_size
    return np.convolve(x, win, 'valid')


def visualize(
        ft_s, axes,
        itv: bool,
        plot_scale: str,  # 'lg'/'ori'/'normal'
        ss_max, c, r=None,
        text_x=None, text_y=None,
        ss_min=None, score_list=None,
        model=None,  # use partial to assign model
        title=None
):
    font_text = {'size': 17}
    font_formula = fm.FontProperties(
        math_fontfamily='cm', size=19)
    font_legend = {'size': 14, 'math_fontfamily': 'cm'}
    ft, s, real_ind, length = pack_test(ft_s)
    with torch.no_grad():
        ft = rnn.pad_sequence(ft, batch_first=True).cuda()
        pred, score = model(ft, length)
    if score_list is not None:
        score_list.append(score.cpu().squeeze().numpy())
    pred = pred[0, :length[0], [0]]  # normal
    plt.subplots_adjust(hspace=0.338)
    if plot_scale == 'ori':
        pred = pred.cpu().numpy()
    else:
        if plot_scale == 'lg':
            s = np.log10(s)
            pred = torch.log10(pred)
        if itv:
            pred = human_intervene(
                pred.unsqueeze(0).cuda(), ss_max,
                torch.from_numpy(real_ind), False, ss_min
            ).squeeze(0).cpu().numpy()
        else:
            if plot_scale == 'normal':
                pred = torch.clamp(pred, 1e-7, 1).squeeze(0).cpu().numpy()
            else:  # lg
                pred = torch.clamp(pred, max=0).squeeze(0).cpu().numpy()
    t = np.arange(length[0])
    ax = axes[c] if r is None else axes[r, c]
    ax.scatter(
        t[real_ind], s[real_ind],
        marker='o', facecolor='white', s=90,
        color='r', label=r'ground truth $s^\mathrm{grd}$'
    )
    ax.scatter(
        t[real_ind], pred[real_ind], s=90,
        marker='v', label=r'predicted result of points $\hat{s}^{\mathrm{grd}}$'
    )
    ax.plot(t, pred, '+', markersize=3,
            label=r'predicted result of interpolated labels $\hat{s}^{\mathrm{itp}}$')
    ax.set_xlabel('Time (s)', fontdict=font_text)
    if plot_scale == 'ori':
        y_label = 'ori'
    elif plot_scale == 'normal':
        y_label = r'$s$'
    else:
        y_label = r'$\lg{s}$'
    ax.set_ylabel(y_label, fontproperties=font_formula)
    # tick specification
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # set tick labels
    # xticks = ax.get_xticks()
    # ticks = np.arange(0, length[0] + 1, ticks_dict[length[0]])
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticks.astype(int), fontdict=font_tick)
    # yticks = ax.get_yticks()
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticks.round(2), fontdict=font_tick)
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    if bool(text_x):
        try:
            r2 = r2_score(s[real_ind], pred[real_ind])
            ax.text(text_x, text_y, f'$R^2={r2:.4f}$',
                    fontproperties=font_formula, horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
        except ValueError:
            ax.set_text(text_x, text_y, f'lg(negative)', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
    ax.set_title(
        title, fontdict=font_text,
        # y=-0.35
    )
    return score_list


def kd2s(kd):
    s = 1 / np.cumprod(1 + kd)
    s = np.hstack(([1], s))
    return s


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class MyPGD:
    def __init__(self, model):
        self.model = model
        self.input_backup = None
        # self.grad_backup = None

    def attack(self, inp: torch.Tensor, epsilon=1., alpha=0.3, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        if is_first_attack:
            self.input_backup = inp.clone()
        norm = torch.norm(inp.grad)
        if norm != 0 and not torch.isnan(norm):
            r_at = alpha * inp.grad / norm
            inp.data.add_(r_at)
            inp.data = self.project(inp.data, epsilon)

    # def restore(self):
    #     .data = self.emb_backup[name]
    #     self.emb_backup = {}

    def project(self, inp_data, epsilon):
        r = inp_data - self.input_backup
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.input_backup + r

    # def backup_grad(self, inp):
    #     self.grad_backup = inp.grad.clone()

    # def restore_grad(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             param.grad = self.grad_backup[name]


if __name__ == '__main__':
    ...
