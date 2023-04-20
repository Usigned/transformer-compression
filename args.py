from model import *
from quant_utils import *
from argparse import Namespace


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


ENV = Namespace(
    **{
        "ori_acc": 0.9853000044822693,
        "lat_b": 1000.,
        "e_b": 1000.,
        "mem_b": 1000.,
        "max_bit": 8,
        "min_bit": 4,
        "a_bit": 8,
        "max_heads": 12,
        "min_heads": 3,
        'head_dim': 64
    }
)


TRAIN_AGENT = Namespace(
    **{
        'tau': 0.01,
        'init_delta': 0.5,
        'delta_decay': 0.99,
        'epsilon': 5000, 
        'warmup': 20,
        'discount': 1.,
        'hidden1': 300,
        'hidden2': 300,
        'lr_a': 3e-4,
        'lr_c': 3e-3,
        'rmsize': 10000,
        'n_update': 3,
        'window_length': 1,
        'bsize': 64
    }
)