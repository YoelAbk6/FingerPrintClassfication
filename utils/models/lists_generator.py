import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

optimizers_init = {"momentum": ['SGD'],
                   "no_momentum": ['Adam', 'Adagrad'], }


def get_models_list():
    return [('Resnet50', models.resnet50),
    ('VGG-19', models.vgg19),
    ('Mobilenet-v2', models.mobilenet_v2), ]


def get_losses_list():
    return [('CrossEntropyLoss', nn.CrossEntropyLoss()), ]


def get_optimizers_list():
    return [('Adam', optim.Adam),
            ('SGD', optim.SGD),
            ('Adagrad', optim.Adagrad)]


def get_learning_rates_list():
    return [('0.0001', 0.0001)]


def get_data_sets_list():
    return [("NIST302a-M", 'data/NIST302/auxiliary/flat/M/500/plain/png/equal'),
            ("SOCOfing", 'data/SOCOFing/Real'),
            ("NIST4", 'data/sd04/png_txt/figs')]
