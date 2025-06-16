import os
import random
from typing import Union
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore")
import medical.models
import medical.datas
from medical.metrics import classification
from medical.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn, print_only
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from monai.metrics import compute_hausdorff_distance
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from medical.utils import print_only
from sklearn.metrics import jaccard_score

parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="",     # Please specify the path to the config file
                    help="Full path to save best validation model")


compute_metrics = ["si_sdr", "sdr"]
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main(config):
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress), 
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )
    config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
         os.path.dirname(__file__), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")
    
    model =  getattr(medical.models, config["train_conf"]["model"]["model_name"]).from_pretrain(
        model_path,
        **config["train_conf"]["model"]["model_config"],
    )

    if config["train_conf"]["training"]["gpus"]:
        device = "cuda"
        model.to(device)
    model_device = next(model.parameters()).device
    datamodule: object = getattr(medical.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets       # Get test dataset
   
    ex_save_dir = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)

    est_labels = []     # pred
    tar_labels = []     # label

    model.eval()
    torch.no_grad().__enter__()
    with progress:
        for idx in progress.track(range(len(test_set))):
            
            datas = tensors_to_device(test_set[idx],device=model_device)
            
            img = datas["image"].unsqueeze(0)             
            label = datas["label"].unsqueeze(0)         

            # est_label = model(img)
            _, est_label = model(img)
                   
            est_labels.append((est_label >= 0.5).long().cpu().numpy())      
            tar_labels.append(label.long().cpu().numpy())                   

    metrics_dict = {}

    print_only("Finished Testing, Start Calculating Metrics")

    tar_labels_flat = np.array(tar_labels).ravel()
    est_labels_flat = np.array(est_labels).ravel()

    metrics_dict['accuracy_score'] = accuracy_score(tar_labels_flat, est_labels_flat)     # (N, C, H, W)
    print_only(f"Accuracy score: {metrics_dict['accuracy_score']}")
    metrics_dict['precision_score'] = metrics.precision_score(tar_labels_flat, est_labels_flat)
    print_only(f"Precision score: {metrics_dict['precision_score']}")
    metrics_dict['recall_score'] = metrics.recall_score(tar_labels_flat, est_labels_flat)
    print_only(f"Recall score: {metrics_dict['recall_score']}")
    metrics_dict['f1_score'] = metrics.f1_score(tar_labels_flat, est_labels_flat)
    print_only(f"F1 score: {metrics_dict['f1_score']} ")

    # ============= Compute HD95 ============= #
    hd95_list = []
    for est, tar in zip(est_labels, tar_labels):
        # import pdb; pdb.set_trace()
        est_tensor = torch.tensor(est)
        tar_tensor = torch.tensor(tar)
        # import pdb; pdb.set_trace()
        if est_tensor.sum() == 0 and tar_tensor.sum() == 0:
            hd95_list.append(0)     # If both are background, HD95 is 0
        elif est_tensor.sum() == 0 or tar_tensor.sum() == 0:
            continue                # hd95 only compute for foreground
        else:
            hd95 = compute_hausdorff_distance(
                y_pred=est_tensor.float(),  # Add channel dimension
                y=tar_tensor.float(),  # Add channel dimension
                include_background=False,
                distance_metric="euclidean",
                percentile=95,
                spacing=None,           # Spacing is None, default to pixel units
            )
            hd95_list.append(hd95.item())

    # import pdb; pdb.set_trace()
    metrics_dict['HD95'] = sum(hd95_list) / len(hd95_list)  # Compute mean HD95
    print_only(f"HD95: {metrics_dict['HD95']}")
    # import pdb; pdb.set_trace()

    # ======================================== #

    with open(ex_save_dir+"/result.json","w") as f:
        json.dump(metrics_dict,f)
          
    print_only("Finished Calculating Metrics.")
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # conf_dir:str, train_conf:dict
    # print(arg_dic)
    main(arg_dic)
