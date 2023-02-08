import torchvision
import torch.nn as nn
import torch.optim as optim

def get_models_list():
    return [('resnet50', torchvision.models.resnet50()), ]

def get_losses_list():
    return [('CrossEntropyLoss', nn.CrossEntropyLoss()), ]

def get_optimizers_list():
    return [('SGD', optim.SGD), ]

def get_learning_rates_list():
    return [('0.001', 0.001), ]