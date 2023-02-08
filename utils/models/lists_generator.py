import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def get_models_list():
    return [('Resnet50', models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)),
            ('VGG-19', models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)),
            ('Mobilenet v2', models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)),]

def get_losses_list():
    return [('CrossEntropyLoss', nn.CrossEntropyLoss()), ]

def get_optimizers_list():
    return [('SGD', optim.SGD), ]

def get_learning_rates_list():
    return [('0.001', 0.001), ]