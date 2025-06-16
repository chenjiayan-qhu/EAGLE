
import torch.nn as nn
from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss
)
from .dice_loss import  DiceLoss
from .gt_bce_dice_loss import GT_BceDiceLoss
from .gt_bce_dice_loss import GT_BceDiceLoss4

__all__ = [
    "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    "DiceLoss",
    "GT_BceDiceLoss",
    "GT_BceDiceLoss4",
]


def make_loss(loss_name="CrossEntropyLoss", **kwargs):
    return get(loss_name)(**kwargs)


def register_loss(custom_opt):
    if (
        custom_opt.__name__ in globals().keys()
        or custom_opt.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Activation {custom_opt.__name__} already exists. Choose another name."
        )
    globals().update({custom_opt.__name__: custom_opt})


def get(identifier):
    if isinstance(identifier, nn.Module):
        return identifier
    elif isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret Loss : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret Loss : {str(identifier)}")