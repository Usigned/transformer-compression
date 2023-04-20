import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LinearGeneral, CAFIA_Transformer
from utils import freeze_but_type, assert_model_all_freezed, assert_all_module_type_freezed, set_freeze_all
import os

class LearnableMask(nn.Module):
    def __init__(self, dim, prune_rate=1.) -> None:
        super().__init__()
        self.dim = dim
        self.pruned_dim = 0
        self.mask = nn.Parameter(torch.ones(dim))
        self.prune_rate = prune_rate
        self._mask = nn.Parameter(torch.ones_like(self.mask), requires_grad=False)
        self.is_fixed = False
        self.random_mask = False

    def forward(self, x):
        mask = self.get_mask()
        x = torch.matmul(x, torch.diag(mask))
        return x, mask

    def get_mask(self):
        if self.random_mask:
            return self._mask
        mask = self.mask if self.is_fixed else self.prune_rate * self.dim * torch.softmax(self.mask, 0)
        return torch.mul(mask, self._mask)

    def fix_mask(self, ratio=1.0):
        self.mask.requires_grad = False
        self.prune_rate = ratio
        self.mask = nn.Parameter(self.get_mask())
        self.is_fixed = True
        return self.mask

    def _make_prune_mask(self, rest_dim):
        assert rest_dim > 0
        self.random_mask = False
        if not self.is_fixed:
            self.fix_mask()
        _mask = torch.ones_like(self._mask, requires_grad=False)
        self.pruned_dim = self.dim - rest_dim
        _, idxs = torch.sort(self.mask)

        for idx in idxs[:self.pruned_dim]:
            _mask[idx] = 0
        self._mask = nn.Parameter(_mask, requires_grad=False).to(device=_mask.device)
    
    def make_prune_mask(self, prune_rate):
        self._make_prune_mask(int(prune_rate*self.dim))


    def make_random_prune_mask(self, prune_rate):
        self.random_mask = True
        self.pruned_dim = self.dim - int(prune_rate*self.dim)
        _mask = torch.cat((torch.ones(int(prune_rate*self.dim)), torch.zeros(self.pruned_dim)), 0)
        self._mask = nn.Parameter(_mask[torch.randperm(self.dim)])

    def extra_repr(self):
        s = "p={}, fixed={}, dim={}".format(self.prune_rate, self.is_fixed, self.dim-self.pruned_dim)
        return s

def get_mask_idx(mvit:CAFIA_Transformer):
    idx = []
    for i, m in enumerate(mvit.modules()):
        if type(m) is LearnableMask:
            idx.append(i)
    return idx

def fix_dmask(mvit:nn.Module):
    for m in mvit.modules():
        if type(m) is LearnableMask:
            m._mask.requires_grad = False

def to_mask(mvit:nn.Module, device):
    for m in mvit.modules():
        if type(m) is LearnableMask:
            m._mask = m._mask.to(device)

def fix_mask(model:CAFIA_Transformer, ratio=1.):
    for m in model.modules():
        if type(m) is LearnableMask:
            m.fix_mask(ratio)

def prune(model:CAFIA_Transformer, prune_rate=1., random=False):
    # fix_mask(model)
    set_mask_prune_rate(model, prune_rate) #只是为了展示，不会影响推理
    for m in model.modules():
        if type(m) is LearnableMask:
            if random:
                m.make_random_prune_mask(prune_rate)
            else:
                m.make_prune_mask(prune_rate)

def mix_prune(model:nn.Module, prune_idx, strategy):
    assert len(strategy) == len(prune_idx)
    i = 0
    for idx, m in enumerate(model.modules()):
        if idx in prune_idx:
            m:LearnableMask
            m._make_prune_mask(strategy[i])
            i += 1


def set_mask_prune_rate(model:CAFIA_Transformer, prune_rate):
    for m in model.modules():
        if type(m) is LearnableMask:
            m.prune_rate=prune_rate

class MaskedSelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1, prune_rate=.25, linear_general=LinearGeneral, head_dim=None):
        super(MaskedSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads if not head_dim else head_dim
        self.scale = self.head_dim ** 0.5
        self.prune_rate = prune_rate

        self.query = linear_general((in_dim,), (self.heads, self.head_dim))
        self.key = linear_general((in_dim,), (self.heads, self.head_dim))
        self.value = linear_general((in_dim,), (self.heads, self.head_dim))
        self.out = linear_general((self.heads, self.head_dim), (in_dim,))

        self.mask = LearnableMask(self.heads, self.prune_rate)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, need_mask=False):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))  # b,n,heads,head_dim
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)  # b,heads,n,head_dim

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)  # b,heads,n,n
        out = torch.matmul(attn_weights, v)  # b, heads, n, head_dim
        out = out.permute(0, 2, 3, 1)  # b, n, head_dim, heads
        out, mask = self.mask(out)
        out = out.permute(0, 1, 3, 2)  # b, n, heads, head_dim
        out = self.out(out, dims=([2, 3], [0, 1]))

        return (out, mask) if need_mask else out

    def get_mask(self):
        return self.mask.get_mask()


def build_mvit(args):
    args.attn_type = MaskedSelfAttention
    args.vit_model = None
    if not hasattr(args, 'pr'):
        args.pr = 1.0
    mvit = CAFIA_Transformer(args)
    if hasattr(args, 'cifar10_vit'):
        if hasattr(args, 'pwd'):
            args.cifar10_vit = os.path.join(args.pwd, args.cifar10_vit)
        mvit = load_weight_for_mvit(mvit, path=args.cifar10_vit)
        print(f"load pretrained weight from {args.cifar10_vit}.")
    else:
        print(f"no pretrained weight loaded cause no path is specified.")
    set_mask_prune_rate(mvit, args.pr)
    return mvit


def test_mmsa():
    b, n, in_dim, heads = 64, 32, 888, 8
    in_shape = (b, n, in_dim)
    mmsa = MaskedSelfAttention(in_dim=in_dim, heads=heads)
    out, mask = mmsa(torch.randn(*in_shape), need_mask=True)
    print(mask)
    assert out.shape == (b, n, in_dim), \
        f'test failed expected out dim {in_shape}, got {out.shape}'


def test_cafia_tfm():
    import json
    from argparse import Namespace
    args = Namespace(
        **json.load(open('Vision-Transformer-ViT/ViT-B_16-224.json', 'r')))
    args.attn_type = MaskedSelfAttention
    args.vit_model = None

    vit = CAFIA_Transformer(args)

    b, c, h, w = (args.batch_size, 3, args.image_size, args.image_size)

    out = vit(torch.randn(*(b, c, h, w)))
    assert out.shape == (b, args.num_classes), \
        f'test failed expected out dim {(b, args.num_classes)}, got {out.shape}'


def test_mvit_bp():
    import json
    from argparse import Namespace
    args = Namespace(
        **json.load(open('Vision-Transformer-ViT/ViT-B_16-224.json', 'r')))
    args.attn_type = MaskedSelfAttention
    args.vit_model = None

    device = 'cuda:0'

    vit = CAFIA_Transformer(args).to(device)

    b, c, h, w = (args.batch_size, 3, args.image_size, args.image_size)

    out = vit(torch.randn(*(b, c, h, w)).to(device))
    label = torch.randn(b, args.num_classes).to(device)

    loss = F.cross_entropy(out, label)
    loss.backward()
    print(loss)


def get_masks_from_mmsa(module: nn.Module):
    masks = []
    for child in module.children():
        if type(child) is LearnableMask:
            masks += [child.mask]
        else:
            masks += get_masks_from_mmsa(child)
    return masks


def get_mask_val_from_masks(module: nn.Module):
    '''
    use get_mask() fn to get mask val during inference
    '''
    masks = []
    for child in module.children():
        if type(child) is LearnableMask:
            masks += [child.get_mask()]
        else:
            masks += get_mask_val_from_masks(child)
    return masks


def freeze_model_but_mask(model: CAFIA_Transformer):
    return freeze_but_type(model, LearnableMask)


def _test_freeze():
    import json
    from argparse import Namespace
    args = Namespace(
        **json.load(open('Vision-Transformer-ViT/ViT-B_16-224.json', 'r')))
    model = build_mvit(args)

    # set_freeze_all(model, True)
    freeze_model_but_mask(model)
    assert_model_all_freezed(model, True, LearnableMask)
    assert_all_module_type_freezed(model, LearnableMask, freeze=False)


def load_weight_for_mvit(model: CAFIA_Transformer, path: str):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    return model

def show_dmask(mvit):
    for m in mvit.modules():
        if type(m) is LearnableMask:
            print(m._mask)

if __name__ == '__main__':
    # test_mmsa()
    # test_cafia_tfm()
    # test_mvit_bp()
    # _test_freeze()
    pass
    # lm = LearnableMask(dim=12)
    # lm.mask = nn.Parameter(torch.randn(12))
    # freeze_but_type(lm, LearnableMask)
    # print(lm._mask.requires_grad)
    # lm.make_random_prune_mask(0.5)
    # lm(torch.randn(1, 12))
