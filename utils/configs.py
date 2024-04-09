"""
parse config file
"""

import os
import random

import configargparse
import namegenerator
import torch
import yaml
from torch.backends import cudnn


def parse_configs():
    # parse args/config file
    parser = configargparse.ArgParser(default_config_files=["./config.yml"])
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="model architecture",
        choices=["resnet18", "resnet50", "convnext", "vit", "simplemlp", "simplecnn"],
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint file, omit for no checkpoint",
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        default="./config.yml",
        help="config file location",
    )
    parser.add_argument("--crop-size", type=int, default=256, help="crop size")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to use")
    parser.add_argument("-r", "--dataset-root", type=str, default="./data/", help="dataset filepath")
    parser.add_argument("--device", type=str, default="cuda", help="gpu(s) to use")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs to train for")
    parser.add_argument("--fold-num", type=int, default=0, help="fold number for cross-validation")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer momentum")
    parser.add_argument("--multiplier", type=int, default=1, help="batch multiplier for SimCLR")
    parser.add_argument("--name", type=str, default="random", help="run name")
    parser.add_argument("--no-tqdm", action="store_true", help="disable tqdm progress bar")
    parser.add_argument("--num-classes", type=int, default=14, help="number of classes in dataset")
    parser.add_argument("--resize-size", type=int, default=256, help="resize size for transform")
    parser.add_argument("--root", type=str, default="runs", help="root of folder to save runs in")
    parser.add_argument("-S", "--seed", type=int, default=-1, help="random seed, -1 for random")
    parser.add_argument("--skip-train", action="store_true", help="skip training")
    parser.add_argument("--transforms", type=str, default="randomcrop", nargs="+")
    parser.add_argument(
        "--use-train-transf-for-val",
        action="store_true",
        help="use the training transformations during the validation phase",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-9, help="optimizer weight decay")
    parser.add_argument("--workers", type=int, default=2, help="dataloader worker threads")

    configs, _ = parser.parse_known_args()

    #########################################
    ## SET UP SEEDS AND PRE-TRAINING FILES ##
    #########################################
    if configs.name == "random":
        configs.name = namegenerator.gen()
    else:
        configs.name = configs.name

    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        cudnn.deterministic = True

    print(f"Run name: {configs.name}")
    os.makedirs(f"{configs.root}/{configs.name}", exist_ok=True)
    configs.root = f"{configs.root}/{configs.name}"

    # save configs object as yaml
    with open(os.path.join(configs.root, "config.yml"), "w", encoding="utf-8") as file:
        yaml.dump(vars(configs), file)

    return configs


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def entropy(logits):
    """compute entropy of logits"""
    # use nansum because 0 predictions give nan values (and log(0) = nan)
    return -torch.nansum(logits * torch.log2(logits), dim=1)
