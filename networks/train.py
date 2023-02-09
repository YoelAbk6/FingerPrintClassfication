import torch
import torch.nn as nn
from utils.models.lists_generator import optimizers_init
import torchvision.models as models


def init_model(model, model_name, device, num_classes):
    out_model = None
    if model_name == 'Resnet50':
        out_model = model(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == 'VGG-19':
        out_model = model(weights=models.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'Mobilenet v2':
        out_model = model(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        raise Exception(
            f'Need to init {model_name} in networks.train.init_model() function!\n')

    out_model = nn.DataParallel(out_model)

    if hasattr(out_model.module, 'classifier'):
        num_features = out_model.module.classifier[-1].in_features
        out_model.module.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        num_features = out_model.module.fc.in_features
        out_model.module.fc = nn.Linear(num_features, num_classes)

    out_model.to(device)
    return out_model


def init_optimizer(optim, name, model, lr=0.001, momentum=0.9):
    if name in optimizers_init.get('momentum'):
        return optim(model.parameters(), lr=lr, momentum=momentum)
    elif name in optimizers_init.get('no_momentum'):
        return optim(model.parameters(), lr=lr)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 100*correct
