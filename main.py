# pylint: disable=used-before-assignment, redefined-outer-name
"""
combination multimodal method

jakob johnson, 4/02/2024
"""
import os
import random

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import models
from tqdm import tqdm

from utils import (
    ConvNeXt,
    ResNet18,
    ResNet50,
    SimpleMLP,
    ViT,
    get_datasets,
    parse_configs,
)

LOGIT_MASKS = {
    0: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U3O8
    1: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # UO2
    2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # UO3
}

# logit masks for true best case
LOGIT_MASKS_2 = {
    0: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U3O8
    1: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U3O8
    2: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U3O8
    3: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U3O8
    4: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # UO2
    5: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # UO2
    6: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # UO2
    7: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # UO2
    8: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # UO2
    9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # UO3
    10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # UO3
    11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # UO3
    12: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # UO3
}


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def train_loop(dataloader, image_model, xrd_model, model, optimizer):
    """Train the model for one epoch."""
    # set model to train mode
    model.train()

    # set up loss and metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(configs.device)

    num_batches = len(dataloader)
    train_loss = 0

    tbar_loader = tqdm(dataloader, desc="train", dynamic_ncols=True, disable=configs.no_tqdm)

    for xrds, sems, labels in tbar_loader:
        # move images to GPU if needed
        xrds, sems, labels = (
            xrds.to(configs.device),
            sems.to(configs.device),
            labels.to(configs.device),
        )

        # zero gradients from previous step
        optimizer.zero_grad()

        # get image representation and XRD token
        _, sem_features = image_model(sems, return_feature=True)
        _, xrd_features = xrd_model(xrds, return_feature=True)

        # concatenate features
        features = torch.cat((sem_features, xrd_features.squeeze()), dim=1)

        # compute prediction and loss
        logits = model(features)
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


def val_loop(val_dataloader, image_model, xrd_model, model):
    """Validate the model for one epoch."""
    # set model to eval mode
    model.eval()

    # set up loss and metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(configs.device)

    num_batches = len(val_dataloader)
    val_loss = 0.0
    incorrect = 0
    with torch.no_grad():
        # validate on in-distribution data
        tbar_loader = tqdm(val_dataloader, desc="val", dynamic_ncols=True, disable=configs.no_tqdm)

        for xrds, sems, labels in tbar_loader:
            # move images to GPU if needed
            xrds, sems, labels = (
                xrds.to(configs.device),
                sems.to(configs.device),
                labels.to(configs.device),
            )

            # get image representation and XRD token
            sem_logits, sem_features = image_model(sems, return_feature=True)
            xrd_logits, xrd_features = xrd_model(xrds, return_feature=True)

            # concatenate features
            features = torch.cat((sem_features, xrd_features.squeeze()), dim=1)

            # compute prediction and loss
            logits = model(features)

            if configs.use_logit_masking_baseline:
                # get class pred from XRD model
                xrd_preds = torch.argmax(F.softmax(xrd_logits.squeeze(), dim=1), dim=1)

                # # get class pred from label (for true best case)
                # xrd_preds = labels

                # make mask for image_logits with class pred
                for i, (logit, pred) in enumerate(zip(sem_logits, xrd_preds)):
                    mask = torch.tensor(LOGIT_MASKS[pred.item()]).to(configs.device)
                    # mask = torch.tensor(LOGIT_MASKS_2[labels[i].item()]).to(configs.device)
                    sem_logits[i] = logit * mask

                logits = sem_logits

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
    configs.dataset = "paired"

    ####################
    ## SET UP DATASET ##
    ####################

    # get image datasets
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
        case "resnet18":
            image_model = ResNet18(num_classes=NUM_CLASSES)
            # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        case "resnet50":
            image_model = ResNet50(num_classes=NUM_CLASSES)
            # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        case "convnext":
            image_model = ConvNeXt()
            image_model.load_state_dict(models.ConvNeXt_Small_Weights.IMAGENET1K_V1.get_state_dict())
            if configs.dataset != "imagenet":
                image_model.classifier[2] = nn.Linear(image_model.classifier[2].in_features, NUM_CLASSES)
        case "vit":
            image_model = ViT()
            image_model.load_state_dict(models.ViT_L_16_Weights.IMAGENET1K_V1.get_state_dict())
            if configs.dataset != "imagenet":
                image_model.heads.head = nn.Linear(image_model.heads.head.in_features, NUM_CLASSES)

    # load checkpoint
    image_model.load_state_dict(torch.load(configs.checkpoint, map_location="cpu"))
    image_model.to(configs.device)

    # freeze model
    image_model.eval()
    for param in image_model.parameters():
        param.requires_grad = False

    # set up XRD model
    xrd_model = SimpleMLP(input_dim=4096, num_classes=3)

    # load checkpoint
    xrd_model.load_state_dict(torch.load(configs.xrd_checkpoint, map_location="cpu"))
    xrd_model.to(configs.device)

    # freeze model
    xrd_model.eval()
    for param in xrd_model.parameters():
        param.requires_grad = False

    # set up classifier model
    model = nn.Sequential(
        nn.Linear(2048 + 16, 2048 + 16),
        nn.ReLU(),
        nn.Linear(2048 + 16, NUM_CLASSES),
    )
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
            train_stats = train_loop(train_dataloader, image_model, xrd_model, model, optimizer)
            val_stats = val_loop(val_dataloader, image_model, xrd_model, model)
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
    test_stats = val_loop(val_dataloader, image_model, xrd_model, model)
    print(f"test acc: {test_stats['val_acc']*100:.2f}%, test loss: {test_stats['val_loss']:.4f}")

    mlflow.log_metrics(test_stats, step=configs.epochs)

    ####################
    ## LOG BEST MODEL ##
    ####################
    model.to("cpu")
    mlflow.pytorch.log_model(model, "best-model", conda_env="env.yaml")
    mlflow.log_artifacts(configs.root)
