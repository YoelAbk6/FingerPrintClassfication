import torch
import torch.nn as nn
from utils.models.lists_generator import optimizers_init
import torchvision.models as models
import torch
import numpy as np
import random

torch.manual_seed(1997)
np.random.seed(1997)
random.seed(1997)


def init_model(model, model_name, device, num_classes):
    out_model = None
    if model_name == 'Resnet50':
        out_model = model(
            weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif model_name == 'VGG-19':
        out_model = model(
            weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    elif model_name == 'Mobilenet-v2':
        out_model = model(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2).to(device)
    elif model_name == 'Resnet101':
        out_model = model(
            weights=models.ResNet101_Weights.IMAGENET1K_V2).to(device)
    else:
        raise Exception(
            f'Need to init {model_name} in networks.train.init_model() function!\n')

    out_model = nn.DataParallel(out_model)

    if hasattr(out_model.module, 'classifier'):
        num_features = out_model.module.classifier[-1].in_features
        out_model.module.classifier[-1] = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.55),
            nn.Linear(128, 2)).to(device)
    else:
        out_model.module.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.55),
            nn.Linear(128, 2)).to(device)
        # num_features = out_model.module.fc.in_features
        # out_model.module.fc = nn.Linear(num_features, num_classes)

    out_model.to(device)
    return out_model


def init_optimizer(optim, name, model, lr=0.001, momentum=0.9):
    if name in optimizers_init.get('momentum'):
        return optim(model.parameters(), lr=lr, momentum=momentum)
    elif name in optimizers_init.get('no_momentum'):
        return optim(model.parameters(), lr=lr)


def train_loop(dataloader, model, loss_fn, optimizer, device, l2_lambda=0.0005):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct, train_loss = 0, 0
    for batch, (X, y, paths) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = y.to(device)
        loss = loss_fn(pred, y)
        # Add L2 regularization
        l2_reg = None
        for param in model.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        loss = loss + l2_lambda * l2_reg
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    train_loss /= num_batches
    return 100 * correct, train_loss


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for X, y, paths in dataloader:
            y_true.extend(y.data.numpy())
            y = y.to(device)
            pred = model(X)
            y_pred.extend(torch.argmax(pred, 1).data.cpu().numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 100 * correct, y_pred, y_true, test_loss
