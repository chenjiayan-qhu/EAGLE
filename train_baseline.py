import os
import sys
import torch
from torch import Tensor
import argparse
import json
import medical.datas
import medical.models
import medical.systems
import medical.utils
import medical.losses
import medical.systems
from medical.losses import make_loss
from medical.systems import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from rich import print, reconfigure
from collections.abc import MutableMapping
from medical.utils import print_only, MyRichProgressBar, RichProgressBarTheme
import monai
import pdb



import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/eagle.yml",                     # Please specify the path to the configuration file xxx.yml
    help="Full path to save best validation model",
)

# ============== Set WanDB API Key ============== #
# os.environ["WANDB_API_KEY"] = ""

def main(config):
    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule: object = getattr(medical.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()
    train_loader, val_loader = datamodule.make_loader
    
    # Define model and optimizer
    print_only(
        "Instantiating Net <{}>".format(config["model"]["model_name"])
    )
    model = getattr(medical.models, config["model"]["model_name"])(
        **config["model"]["model_config"],
    )
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        scheduler = getattr(medical.systems.lr_schedulers, config["scheduler"]["sche_name"])(
            optimizer=optimizer, **config["scheduler"]["sche_config"]
        )

    # Just after instantiating, save the args. Easy loading in the future.  
    # config["main_args"]["exp_dir"] = os.path.join(
    #     os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    # ) 
    config["main_args"]["exp_dir"] = os.path.join(
        os.path.dirname(__file__), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]        # 实验路径
    os.makedirs(exp_dir, exist_ok=True)             # 确保路径存在
    conf_path = os.path.join(exp_dir, "conf.yml")   # 配置文件路径
    with open(conf_path, "w") as outfile:           # 保存到配置文件中
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>".format(
            config["loss"]["loss_name"]
        )
    )
    loss_func = make_loss(loss_name=config["loss"]["loss_name"])

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(medical.systems, config["training"]["system"])(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}-{val_acc:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "gpu" if torch.cuda.is_available() else None

    tensorboard_logger = TensorBoardLogger(
        os.path.join(os.path.dirname(__file__), "Experiments", "tensorboard_logs"),
        name=config["exp"]["exp_name"],
    )
    # Setup WanDB logger
    wandb_logger = WandbLogger(
        project="EAGLE",
        name=config["exp"]["exp_name"],
        save_dir=os.path.join(os.path.dirname(__file__), "Experiments", "wandb_logs"),
        config=config,
    )


    strategy = config["training"]["parallel"]
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        # gpus=gpus,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=strategy,
        limit_train_batches=1.0,  # Useful for fast experiment
        # gradient_clip_val=5.0,
        logger=[tensorboard_logger, wandb_logger],
        sync_batchnorm=True,
        # fast_dev_run=True,
    )
    trainer.fit(system)
    print_only("Finished Training")
    
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    print_only("Model saved to {}".format(os.path.join(exp_dir, "best_model.pth")))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from medical.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)
