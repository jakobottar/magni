# pylint: disable=used-before-assignment, redefined-outer-name
"""
classification model on xrd data

jakob johnson, 4/02/2024
"""
import os

import mlflow
import numpy as np
import pyxis.torch as pxt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from utils import SimpleMLP, get_datasets, parse_configs


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def train_loop(dataloader, model, optimizer):
    """Train the model for one epoch."""
    # set model to train mode
    model.train()

    # set up loss and metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(configs.device)

    num_batches = len(dataloader)
    train_loss = 0

    tbar_loader = tqdm(dataloader, desc="train", dynamic_ncols=True, disable=configs.no_tqdm)

    for data, labels in tbar_loader:
        # move images to GPU if needed

        data, labels = (
            data.to(configs.device),
            labels.to(configs.device),
        )

        # zero gradients from previous step
        optimizer.zero_grad()

        # compute prediction and loss
        logits = model(data)
        loss = loss_fn(logits, labels)
        train_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # update metrics
        accuracy.update(torch.argmax(F.softmax(logits, dim=1), dim=1), labels)

    return {
        "train_acc": float(accuracy.compute()),
        "train_loss": train_loss / num_batches,
        "learning_rate": scheduler.get_last_lr()[0],
    }


def val_loop(val_dataloader, model):
    """Validate the model for one epoch."""
    # set model to eval mode
    model.eval()

    # set up loss and metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(configs.device)

    num_batches = len(val_dataloader)
    val_loss = 0.0
    with torch.no_grad():
        # validate on in-distribution data
        tbar_loader = tqdm(val_dataloader, desc="val", dynamic_ncols=True, disable=configs.no_tqdm)

        for data, labels in tbar_loader:
            # move images to GPU if needed
            data, labels = (data.to(configs.device), labels.to(configs.device))

            # compute prediction and loss
            logits = model(data)
            val_loss += loss_fn(logits, labels).item()
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

            # update metric
            accuracy.update(preds, labels)

    return {
        "val_acc": float(accuracy.compute()),
        "val_loss": val_loss / num_batches,
    }


if __name__ == "__main__":
    # parse args/config file
    configs = parse_configs()

    ####################
    ## SET UP DATASET ##
    ####################

    # get datasets
    datasets = get_datasets(configs)
    NUM_CLASSES = datasets["num_classes"]

    print(datasets["train"])
    print(datasets["val"])

    # set up dataloaders
    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=configs.workers,
    )
    val_dataloader = DataLoader(
        datasets["val"],
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.workers,
    )

    ##################
    ## SET UP MODEL ##
    ##################
    print(f"Using device: {configs.device}")

    # choose model architecture
    match configs.arch.lower():
        case "simplemlp":
            model = SimpleMLP(input_dim=4096, num_classes=NUM_CLASSES)

        case "simplecnn":
            raise NotImplementedError
            # model = nn.Sequential(

    # load checkpoint if provided
    if configs.checkpoint is not None:
        model.load_state_dict(torch.load(configs.checkpoint, map_location="cpu"))

    model.to(configs.device)

    # initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            configs.epochs * len(train_dataloader),
            1,
            1e-6 / configs.lr,
        ),
    )

    #################
    ## TRAIN MODEL ##
    #################
    mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
    mlflow.set_experiment("magni")
    mlflow.start_run(run_name=configs.name)
    mlflow.log_params(vars(configs))

    if not configs.skip_train:
        print("Training model...")

        best_metric = 0
        for epoch in range(configs.epochs):
            print(f"epoch {epoch + 1}/{configs.epochs}")
            train_stats = train_loop(train_dataloader, model, optimizer)
            val_stats = val_loop(val_dataloader, model)
            mlflow.log_metrics(train_stats | val_stats, step=epoch)

            print(
                f"epoch {epoch+1}/{configs.epochs} -- train acc: {train_stats['train_acc']*100:.2f}%, train loss: {train_stats['train_loss']:.4f}, val acc: {val_stats['val_acc']*100:.2f}%"
            )

            # save "best" model
            if val_stats["val_acc"] > best_metric:
                best_metric = val_stats["val_acc"]
                torch.save(model.state_dict(), os.path.join(configs.root, "best.pth"))

            # save last model
            torch.save(model.state_dict(), os.path.join(configs.root, "last.pth"))

    print("Done!")

    #####################
    ## TEST BEST MODEL ##
    #####################

    # load best model
    if not configs.skip_train:
        model.load_state_dict(torch.load(os.path.join(configs.root, "best.pth"), map_location=torch.device("cpu")))
        model.to(configs.device)

    # test best model
    test_stats = val_loop(val_dataloader, model)
    print(f"test acc: {test_stats['val_acc']*100:.2f}%, test loss: {test_stats['val_loss']:.4f}")

    mlflow.log_metrics(test_stats, step=configs.epochs)

    ####################
    ## LOG BEST MODEL ##
    ####################
    model.to("cpu")
    mlflow.pytorch.log_model(model, "best-model", conda_env="env.yaml")
    mlflow.log_artifacts(configs.root)
