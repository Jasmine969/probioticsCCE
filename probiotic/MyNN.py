import math

from torch import nn
import torch
from copy import deepcopy as dc


class MyLSTM(nn.Module):
    def __init__(self, input_dim, lstm_dim, lstm_layer,
                 fc_dim, drop, max_t, threshold=None):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim,
                            num_layers=lstm_layer,
                            batch_first=True, dropout=drop)
        self.fc = nn.Linear(lstm_dim, fc_dim)
        # self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.out_layer = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(fc_dim, 1)
        )
        self.threshold = threshold
        if self.threshold:
            self.t = 0
        self.max_t = max_t

    def forward(self, x, mode):
        x, _ = self.lstm(x)
        x, out_len = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.fc(x)
        x = self.gelu(x)
        out = self.out_layer(x)
        if self.threshold:
            self.t = self.t + 1
            if self.t > self.threshold:
                if self.t - self.threshold == 1 and mode == 'train':
                    print('\033[0;34mIntervention begins!\033[0m')
                out[:, 0, :] = self.max_t  # set the initial value as 1
                out = torch.clamp(out, -1000, self.max_t)
        return out


class PositionEncoding(nn.Module):
    def __init__(self, in_channels, max_len, weight):
        super(PositionEncoding, self).__init__()
        denominator = 10000 ** (torch.arange(0, in_channels, 2) / in_channels)
        pe = torch.zeros(max_len, in_channels)
        pos = torch.arange(max_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)
        pe = pe * weight
        pe.unsqueeze_(0)  # (max_len,d) --> (b,max_len,d)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausalConv, self).__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x):
        """
        :param x: (b,seq,in_channels)
        :return: (b,seq,out_channels)
        """
        out = self.conv(x.transpose(-2, -1))
        dims = out.dim()
        assert dims in [3, 4]
        if dims == 3:
            out = out[:, :, :-self.padding].contiguous()
        else:
            out = out[:, :, :, :-self.padding].contiguous()
        return out.transpose(-2, -1)


class ConvAttention(nn.Module):
    def __init__(
            self, in_channels, kernel_size, dk,
            act_name,  # 'relu','gelu','prelu','srelu','elu'
            dv, num_heads, dropout, pos_weight=None,
            pos_enc=0, out_negative=False,
            verbose=False  # if True, return mat at each node
    ):
        super(ConvAttention, self).__init__()
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.activate = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'prelu': nn.PReLU,
            'leakyrelu': nn.LeakyReLU,
            'srelu': nn.SELU,
            'elu': nn.ELU
        }
        self.verbose = verbose
        if pos_enc:
            self.pos_enc = PositionEncoding(in_channels, 401, pos_weight)
        else:
            self.pos_enc = None
        self.conv_q = CausalConv(
            in_channels=in_channels,
            out_channels=dk * num_heads,
            kernel_size=kernel_size
        )
        self.conv_k = CausalConv(
            in_channels=in_channels,
            out_channels=dk * num_heads,
            kernel_size=kernel_size
        )
        self.w_v = nn.Sequential(
            nn.Linear(in_channels, dv * num_heads),
            self.activate[act_name]()
        )
        self.out_negative = out_negative
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.last_layers = nn.Sequential(
            nn.Linear(dv * num_heads, dv),
            self.activate[act_name](),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(dv, 1)
        )

        self.initialize()

    def initialize(self, layer=None):
        if layer is None:
            for name, param in self.named_parameters():
                if 'weight' in name and param.numel() > 2:
                    nn.init.kaiming_normal_(param)
        else:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.numel() > 2:
                    nn.init.kaiming_normal_(param)

    def split_heads(self, x, dim):
        """
        split multi-head into each single head
        :param x: (b,seq,d*num_heads)
        :param dim: d
        :return: (b,num_heads,seq,d)
        """
        b = x.size(0)
        out = x.reshape(b, -1, self.num_heads, dim)
        return out.transpose(1, 2)

    @staticmethod
    def merge_heads(x):
        """
        merge different single heads into multi-head
        :param x: (b,num_heads,seq,dv)
        :return: (b,seq,num_heads*dv)
        """
        b, _, seq, _ = x.size()
        x = x.transpose(1, 2)
        return x.reshape(b, seq, -1)

    @staticmethod
    def softmax_mask(attn, length):
        """
        future mask --> softmax --> padding mask
        :param length: real seq length
        :param attn: (QK^T)/sqrt(dk), shape:(b,num_heads,seq,seq)
        :return: the same shape as attn, with 0 or 1 as the elements
        """
        future_mask = torch.triu(torch.ones_like(attn), diagonal=1)
        score = torch.softmax(attn + future_mask * -1e9, dim=-1)
        # padding mask
        for ind, each_len in enumerate(length):
            score.data[ind, :, each_len:, :] = 0
        return score

    def forward(self, x, length):
        """
        :param length:
        :param x: (b,seq,in_channels) in_channels=3
        :return: out (b,seq)
        """
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        q = self.split_heads(self.conv_q(x), self.dk)  # (b,num_heads,seq,dk)
        k = self.split_heads(self.conv_k(x), self.dk)  # (b,num_heads,seq,dk)
        v = self.split_heads(self.w_v(x), self.dv)  # (b,num_heads,seq,dv)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        score = self.softmax_mask(score, length)  # (b,num_heads,seq,seq)

        out1 = self.dropout(score) @ v  # (b,num_heads,seq,dv)
        out2 = self.merge_heads(out1)  # (b,seq,num_heads*dv)
        out3 = self.last_layers(out2)  # (b,seq,1)
        if self.out_negative:
            out3 = torch.where(out3 <= 0, out3, 1 - torch.exp(-out3))
        if self.verbose:
            return q, k, score, v, out2, out3
        return out3, score


class ConvAttn2Tower(ConvAttention):
    def __init__(self, in_channels, kernel_size, dk,
                 act_name,
                 dv, num_heads, dropout, pos_weight=None,
                 pos_enc=0, out_negative=False,
                 verbose=False):
        super(ConvAttn2Tower, self).__init__(
            in_channels, kernel_size, dk,
            act_name,
            dv, num_heads, dropout, pos_weight,
            pos_enc, out_negative,
            verbose)
        self.last_layers2 = nn.Sequential(
            nn.Linear(dv * num_heads, dv),
            self.activate[act_name](),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(dv, 1)
        )
        self.initialize(layer=self.last_layers2)
        # self.initialize()

    def forward(self, x, length):
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        q = self.split_heads(self.conv_q(x), self.dk)  # (b,num_heads,seq,dk)
        k = self.split_heads(self.conv_k(x), self.dk)  # (b,num_heads,seq,dk)
        v = self.split_heads(self.w_v(x), self.dv)  # (b,num_heads,seq,dv)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        score = self.softmax_mask(score, length)  # (b,num_heads,seq,seq)

        out0 = self.dropout(score) @ v  # (b,num_heads,seq,dv)
        out0 = self.merge_heads(out0)  # (b,seq,num_heads*dv)
        out1 = self.last_layers(out0)  # (b,seq,1), normal
        out2 = self.last_layers2(out0)  # (b,seq,1), lg
        # lg-->normal no requirement; normal-->lg requires pred>0
        # so use normal as the judgement
        out2 = 10 ** out2  # lg --> normal
        mean_out = (out1 + out2) * 0.5
        w_lg = 1.008 / (1 + 0.009428 * torch.exp(10.78 * mean_out))
        coupled_out = (1 - w_lg) * out1 + w_lg * out2
        out2 = torch.log10(out2)
        return coupled_out, [out1, out2], score


class CoupledModel(nn.Module):
    def __init__(self, normal_net, lg_net):
        super(CoupledModel, self).__init__()
        self.normal_net = normal_net
        self.normal_net.eval()
        self.lg_net = lg_net
        self.lg_net.eval()
        self.cp = 1 / math.log(10)

    def forward(self, x, length, out_last=False):
        normal_output, normal_score = self.normal_net(x, length, out_last)
        lg_output, lg_score = self.lg_net(x, length, out_last)
        lg_output = 10 ** lg_output
        score = torch.cat((normal_score, lg_score), dim=1).mean(dim=1)
        mean_output = (normal_output + lg_output) * 0.5
        w_lg = 1.008 / (1 + 0.009428 * torch.exp(10.78 * mean_output))
        # coupled_output = torch.where(
        #     mean_output >= self.cp, normal_output, lg_output)
        coupled_output = (1 - w_lg) * normal_output + w_lg * lg_output
        return normal_output, lg_output, coupled_output, score


class BaggingModel(nn.Module):
    def __init__(self, model_list: list, vali_acc: list = None):
        super(BaggingModel, self).__init__()
        self.model_num = len(model_list)
        self.model_list = [net.eval() for net in model_list]
        if bool(vali_acc):
            tmp = 1 / (1 - torch.Tensor(vali_acc))
            self.w = (tmp / tmp.sum()).reshape(-1, 1, 1).cuda()
        else:  # arithmetical mean
            self.w = 1 / self.model_num * torch.ones(self.model_num, 1, 1).cuda()

    def forward(self, x, length):
        output_list = [torch.Tensor(0)] * self.model_num
        score_list = [torch.Tensor(0)] * self.model_num
        for i, model in enumerate(self.model_list):
            output_list[i], _, score_list[i] = model(x, length)  # Tower NN has 3 outputs
            score_list[i] = score_list[i].mean(dim=1, keepdim=True)
        output = (torch.cat(output_list, dim=0) * self.w
                  ).sum(dim=0, keepdim=True)
        score = (torch.cat(score_list, dim=0) * self.w.unsqueeze(1)
                 ).sum(dim=0, keepdim=True)
        return output, score


if __name__ == '__main__':
    my_net = BaggingModel([1, 23, 4, 5, 6])
