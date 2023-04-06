from collections.abc import Iterable
import torch.nn as nn

def size(shape):
    size = 1
    for d in shape:
        size *= d
    return size

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
       
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def set_freeze_by_type(model:nn.Module, layer_types, freeze=True):
    if not isinstance(layer_types, Iterable):
        layer_types = [layer_types]
    for child in model.modules():
        if type(child) not in layer_types:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
    return model

def freeze_by_type(model, layer_types):
    return set_freeze_by_type(model, layer_types, True)

def unfreeze_by_type(model, layer_types):
    return set_freeze_by_type(model, layer_types, False)

def set_freeze_all(model:nn.Module, freeze=True):
    for child in model.modules():
        for param in child.parameters():
            param.requires_grad = not freeze
    return model

def freeze_but_type(model, layer_types):
    model = set_freeze_all(model, True)
    return unfreeze_by_type(model, layer_types)


def assert_model_all_freezed(model: nn.Module, freeze=True, exclude_type=None):
    if type(model) is exclude_type:
        return

    if len(list(model.children())) != 0:
        for child in model.children():
            assert_model_all_freezed(child, freeze, exclude_type)
    else:
        for param in model.parameters():
            assert param.requires_grad is not freeze, f"param in {model} has attr requires_grad {param.requires_grad}"


def assert_all_module_type_freezed(model: nn.Module, target_type, freeze=True):
    for module in model.modules():
        if type(module) is target_type:
            assert_model_all_freezed(module, freeze, None)