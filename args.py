from model import *
from quant_utils import *
from argparse import Namespace
from mmsa import MaskedSelfAttention
from data import get_cifar10_dataloader # type: ignore

ALL = Namespace(
    **{
        "learning_rate": 1e-04,
        "opt_eps": None,
        "beta1": 0.99,
        "beta2": 0.99,
        "eps": 1e-06,
        "momentum": 0.9,
        "weight_decay": 2e-05,
        "warmup": 100,
        "batch_size": 32,
        "epoches": 100,
        "output": "./output/mvit",
        "vit_model": "C:\\Users\\1\\Downloads\\imagenet21k+imagenet2012_ViT-B_16-224.pth",
        "image_size": 224,
        "num_classes": 10,
        "patch_size": 16,
        "emb_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "attn_dropout_rate": 0.0,
        "dropout_rate": 0.1,
        "cifar10_vit": "./output/vit/0.9853000044822693.pt",
        "alpha": 1e-02,
    }
)

TRAIN = Namespace(
    **{
        "learning_rate": 1e-04,
        "opt_eps": None,
        "beta1": 0.99,
        "beta2": 0.99,
        "eps": 1e-06,
        "momentum": 0.9,
        "weight_decay": 2e-05,
        "warmup": 100,
        "batch_size": 32,
        "epoches": 100,
        "alpha": 1e-02,
    }
)


VIT = Namespace(
    **{
        "image_size": 224,
        "num_classes": 10,
        "patch_size": 16,
        "emb_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "attn_type": SelfAttention,
        "attn_dropout_rate": 0.0,
        "dropout_rate": 0.1,
        "linear": nn.Linear,
        "linear_general": LinearGeneral,
    }
)

QVIT = Namespace(
    **{
        "image_size": 224,
        "num_classes": 10,
        "patch_size": 16,
        "emb_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "attn_type": SelfAttention,
        "attn_dropout_rate": 0.0,
        "dropout_rate": 0.1,
        "linear": QLinear,
        "linear_general": QLinearGeneral,
    }
)


MQVIT = Namespace(
    **{
        "image_size": 224,
        "num_classes": 10,
        "patch_size": 16,
        "emb_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "attn_type": MaskedSelfAttention,
        "attn_dropout_rate": 0.0,
        "dropout_rate": 0.1,
        "linear": QLinear,
        "linear_general": QLinearGeneral,
    }
)


ENV = {
    # "weight_path": r'/home/ma-user/work/Vision-Transformer-ViT/output/mvit/pat-0.5.pt',
    "weight_path": r'D:\d-storage\output\pat-0.5.pt',
    'trainloader': get_cifar10_dataloader(train=True),
    'testloader': get_cifar10_dataloader(train=False),
    "ori_acc": 0.9853000044822693,
    "max_bit": 8,
    "min_bit": 4,
    "a_bit": 8,
    "max_heads": 12,
    "min_heads": 3,
    'head_dim': 64
}


AGENT = {
    'hidden_dim': 300,
    'action_bound': 1.,
    'gamma': 1.0,
    'actor_lr': 3e-4,
    'critic_lr': 3e-3,
    'tau': 5e-3,
    'init_delta': 0.5,
    'delta_decay': 0.999,
}

COEF_LAT = [1.3257e-18, 1.6811e-03, 2.7591e-05]

COEF_E = 1.0399999999770126

# 0.94s, 0.978j, 48.9MB
__possible_bound_lat_e_m = 940.6659570182138, 978.2925952773189, 48900