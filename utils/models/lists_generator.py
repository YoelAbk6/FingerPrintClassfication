import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import random
import torch
import numpy as np
from utils import definitions

torch.manual_seed(definitions.RANDOM_SEED)
np.random.seed(definitions.RANDOM_SEED)
random.seed(definitions.RANDOM_SEED)

optimizers_init = {"momentum": ['SGD'],
                   "no_momentum": ['Adam', 'Adagrad'], }


def get_models_list():
    return [
        # ('Resnet50', models.resnet50),
        ('Resnet18', models.resnet18),
        #     ('Resnet101', models.resnet101),
        # ('VGG-19', models.vgg19),
        # ('Mobilenet-v2', models.mobilenet_v2)
    ]


def get_losses_list():
    return [('CrossEntropyLoss', nn.CrossEntropyLoss()), ]


def get_optimizers_list():
    return [
        # ('Adam', optim.Adam),
        #     ('SGD', optim.SGD),
        ('Adagrad', optim.Adagrad)
    ]


def get_learning_rates_list():
    return [('0.0001', 0.0001)]


def get_data_sets_list():
    return [
             ("NIST302a-M", 'data/NIST302/auxiliary/flat/M/500/plain/png/regular'),
            # ("NIST302a-M-inner30", 'data/NIST302/auxiliary/flat/M/500/plain/png/inner30'),
            # ("NIST302a-M-inner60", 'data/NIST302/auxiliary/flat/M/500/plain/png/inner60'),
            #("NIST302a-M-4-split-18to28", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/18to28'),
            #("NIST302a-M-4-split-28to38", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/28to38'),
            #("NIST302a-M-4-split-38to48", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/38to48'),
            #("NIST302a-M-4-split-48to58", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/48to58'),
            # ("SOCOfing", 'data/SOCOFing/Real'),
            # ("NIST4", 'data/sd04/png_txt/figs'),
            #("SOCOFing-inner30", 'data/SOCOFing/inner30'),
            #("SOCOFing-inner60", 'data/SOCOFing/inner60'),
            #("NIST4-inner30", 'data/sd04/png_txt/inner30'),
            #("NIST4-inner60", 'data/sd04/png_txt/inner60')
            ]
