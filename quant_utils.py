import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
import math
from model import *

class QModule(nn.Module):
    def __init__(self, w_bit=-1, a_bit=-1, half_wave=True):
        super(QModule, self).__init__()

        if half_wave:
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1
        self._w_bit = w_bit
        self._b_bit = 32
        self._half_wave = half_wave

        self.init_range = 6.
        self.activation_range = nn.Parameter(torch.Tensor([self.init_range]))
        self.weight_range = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)

        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = True
        self._calibrate = False

    @property
    def w_bit(self):
        return self._w_bit

    @w_bit.setter
    def w_bit(self, w_bit):
        self._w_bit = w_bit

    @property
    def a_bit(self):
        if self._half_wave:
            return self._a_bit
        else:
            return self._a_bit + 1

    @a_bit.setter
    def a_bit(self, a_bit):
        if self._half_wave:
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1

    @property
    def b_bit(self):
        return self._b_bit

    @property
    def half_wave(self):
        return self._half_wave

    @property
    def quantized(self):
        return self._quantized

    @property
    def tanh_weight(self):
        return self._tanh_weight

    def set_quantize(self, quantized):
        self._quantized = quantized

    def set_tanh_weight(self, tanh_weight):
        self._tanh_weight = tanh_weight
        if self._tanh_weight:
            self.weight_range.data[0] = 1.0

    def set_fix_weight(self, fix_weight):
        self._fix_weight = fix_weight

    def set_activation_range(self, activation_range):
        self.activation_range.data[0] = activation_range

    def set_weight_range(self, weight_range):
        self.weight_range.data[0] = weight_range

    def set_trainable_activation_range(self, trainable_activation_range=True):
        self._trainable_activation_range = trainable_activation_range
        self.activation_range.requires_grad_(trainable_activation_range)

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_tanh(self, tanh=True):
        self._tanh_weight = tanh

    def _compute_threshold(self, data, bitwidth):
        mn = 0
        mx = np.abs(data).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(data), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        return threshold

    def _quantize_activation(self, inputs):
        if self._quantized and self._a_bit > 0:
            if self._calibrate:
                if self._a_bit < 5:
                    threshold = self._compute_threshold(inputs.data.cpu().numpy(), self._a_bit)
                    estimate_activation_range = min(min(self.init_range, inputs.abs().max().item()), threshold)
                else:
                    estimate_activation_range = min(self.init_range, inputs.abs().max().item())
                # print('range:', estimate_activation_range, '  shape:', inputs.shape, '  inp_abs_max:', inputs.abs().max())
                self.activation_range.data = torch.tensor([estimate_activation_range], device=inputs.device)
                return inputs

            if self._trainable_activation_range:
                if self._half_wave:
                    ori_x = 0.5 * (inputs.abs() - (inputs - self.activation_range).abs() + self.activation_range)
                else:
                    ori_x = 0.5 * ((-inputs - self.activation_range).abs() - (inputs - self.activation_range).abs())
            else:
                if self._half_wave:
                    ori_x = inputs.clamp(0.0, self.activation_range.item())
                else:
                    ori_x = inputs.clamp(-self.activation_range.item(), self.activation_range.item())

            scaling_factor = self.activation_range.item() / (2. ** self._a_bit - 1.)
            x = ori_x.detach().clone()
            x.div_(scaling_factor).round_().mul_(scaling_factor)

            # STE
            # x = ori_x + x.detach() - ori_x.detach()
            return STE.apply(ori_x, x)
        else:
            return inputs

    def _quantize_weight(self, weight):
        if self._tanh_weight:
            weight = weight.tanh()
            weight = weight / weight.abs().max()

        if self._quantized and self._w_bit > 0:
            threshold = self.weight_range.item()
            if threshold <= 0:
                threshold = weight.abs().max().item()
                self.weight_range.data[0] = threshold

            if self._calibrate:
                if self._w_bit < 5:
                    threshold = self._compute_threshold(weight.data.cpu().numpy(), self._w_bit)
                else:
                    threshold = weight.abs().max().item()
                self.weight_range.data[0] = threshold
                return weight

            ori_w = weight

            scaling_factor = threshold / (pow(2., self._w_bit - 1) - 1.)
            w = ori_w.clamp(-threshold, threshold)
            # w[w.abs() > threshold - threshold / 64.] = 0.
            w.div_(scaling_factor).round_().mul_(scaling_factor)

            # STE
            if self._fix_weight:
                # w = w.detach()
                return w.detach()
            else:
                # w = ori_w + w.detach() - ori_w.detach()
                return STE.apply(ori_w, w)
        else:
            return weight

    def _quantize_bias(self, bias):
        if bias is not None and self._quantized and self._b_bit > 0:
            if self._calibrate:
                return bias
            ori_b = bias
            threshold = ori_b.data.max().item() + 0.00001
            scaling_factor = threshold / (pow(2., self._b_bit - 1) - 1.)
            b = torch.clamp(ori_b.data, -threshold, threshold)
            b.div_(scaling_factor).round_().mul_(scaling_factor)
            # STE
            if self._fix_weight:
                return b.detach()
            else:
                # b = ori_b + b.detach() - ori_b.detach()
                return STE.apply(ori_b, b)
        else:
            return bias

    def _quantize(self, inputs, weight, bias):
        inputs = self._quantize_activation(inputs=inputs)
        weight = self._quantize_weight(weight=weight)
        # bias = self._quantize_bias(bias=bias)
        return inputs, weight, bias

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        return 'w_bit={}, a_bit={}, half_wave={}, tanh_weight={}'.format(
            self.w_bit if self.w_bit > 0 else -1, self.a_bit if self.a_bit > 0 else -1,
            self.half_wave, self._tanh_weight
        )


class STE(torch.autograd.Function):
    # for faster inference
    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class QLinear(QModule):
    def __init__(self, in_features, out_features, bias=True, w_bit=-1, a_bit=-1, half_wave=True):
        super(QLinear, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        return F.linear(inputs, weight=weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={w_bit}, a_bit={a_bit}'.format(w_bit=self.w_bit, a_bit=self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s


class QLinearGeneral(QModule):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64), w_bit=-1, a_bit=-1, half_wave=True):
        super(QLinearGeneral, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)

        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

        self.reset_parameters()

    def forward(self, inputs, dims=([2], [0])):
        inputs, weight, bias = inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        return torch.tensordot(inputs, weight, dims=dims) + bias

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = 'in_dim={}, feat_dim={}'.format(
            self.in_dim, self.feat_dim)
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={w_bit}, a_bit={a_bit}'.format(w_bit=self.w_bit, a_bit=self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s

def set_mixed_precision(model:nn.Module, quantizable_idx, strategy):
    assert len(quantizable_idx) == len(strategy), \
        'You should provide the same number of bit setting as layer list for weight quantization!'
    quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        else:
            layer.w_bit = quantize_layer_bit_dict[i][0]
            layer.a_bit = quantize_layer_bit_dict[i][1]

def get_qmvit(args, path=None):
    args.linear = QLinear
    args.linear_general = QLinearGeneral
    args.attn_type = SelfAttention
    qvit = CAFIA_Transformer(args)

    if path:
        load_weight_for_vit(qvit, path)
    return qvit

def get_quantizable_idx(model:nn.Module):
    idx = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, QModule):
            idx.append(i)
    return idx

def get_single_prec_quant_strategy(model:nn.Module, w=8, a=8):
    idx = get_quantizable_idx(model)
    weight_strategy = [w] * len(idx)
    act_strategy = [a] * len(idx)
    return idx, mix_weight_act_strategy(weight_strategy, act_strategy)

def mix_weight_act_strategy(w_s, a_s):
    s = []
    for b1, b2 in zip(w_s, a_s):
        s.append((b1, b2))
    return s