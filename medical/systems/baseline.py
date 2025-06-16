
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from torchmetrics.functional import accuracy
from medical.losses import DiceLoss


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_
    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.
    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class BaselineLightningModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.dice_loss = DiceLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss"
        self.save_hyperparameters(self.config_to_hparams(self.config))

    def forward(self, images):
        """Applies forward pass of the model.
        Returns:
            :class:`torch.Tensor`
        """
        return self.model(images)

    def training_step(self, batch, batch_nb):
        images = batch['image']           # [B, C, H, W]   [0-1]
        tar_labels = batch['label']     # [B, C, H, W]   [0-1]
        # gt_pre, est_labels = self(images)       # forward —> gt_pre, out
        est_labels = self(images)
        # 计算训练的loss
        # loss = self.loss_func(gt_pre, est_labels, tar_labels)       # train_loss
        loss = self.loss_func(est_labels, tar_labels)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}


    def validation_step(self, batch, batch_idx: int):
        # import pdb; pdb.set_trace()
        images = batch['image']
        tar_labels = batch['label']    
        est_labels = self(images)  
        
        # loss for val
        val_loss = self.loss_func(est_labels, tar_labels)
        # dsc for val
        val_dsc = 1. - self.dice_loss(est_labels, tar_labels)
        
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "val_dice",
            val_dsc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
             
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"val_loss": val_loss, "val_dsc": val_dsc}

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.
        Args:
            dic (dict): Dictionary to be transformed.
        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v) # ValueError: too many dimensions 'str'
        return dic