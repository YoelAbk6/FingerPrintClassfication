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
        ('Resnet50', models.resnet50),
        ('Resnet18', models.resnet18),
        ('Resnet101', models.resnet101),
        ('VGG-16', models.vgg16),
        ('VGG-19', models.vgg19),
        ('Mobilenet-v2', models.mobilenet_v2)
    ]


def get_losses_list():
    return [('CrossEntropyLoss', nn.CrossEntropyLoss()), ]


def get_optimizers_list():
    return [
        # ('Adam', optim.Adam),
        # ('SGD', optim.SGD),
        ('Adagrad', optim.Adagrad)
    ]


def get_learning_rates_list():
    return [('0.0001', 0.0001)]


def get_data_sets_list():
    return [
        # ("NIST302a-M", 'data/NIST302/auxiliary/flat/M/500/plain/png/regular'),
        ("NIST302a-easiest-MOC", 'data/NIST302/auxiliary/flat/M/500/plain/png/augmented-confidence-easiest/regular'),
        ("NIST302a-M-best-95-flip-rate-train-only", 'data/NIST302/auxiliary/flat/M/500/plain/png/best-95-flip-rate-train-only/regular'),
        # ("NIST302a-M-worse-flip", 'data/NIST302/auxiliary/flat/M/500/plain/png/worse-95-flip-rate/regular'),
        # ("NIST302a-M-clean", 'data/NIST302/auxiliary/flat/M/500/plain/png/clean_lab/regular'),
        # ("NIST302a-M-inner50-clean", 'data/NIST302/auxiliary/flat/M/500/plain/png/clean_lab/inner50'),
        # ("NIST302a-M-outer50-clean", 'data/NIST302/auxiliary/flat/M/500/plain/png/clean_lab/outer50'),
        # ("NIST302a-M-inner50-confidence",
        #  'data/NIST302/auxiliary/flat/M/500/plain/png/augmented-confidence/inner50'),
        # ("NIST302a-M-outer50-confidence",
        #  'data/NIST302/auxiliary/flat/M/500/plain/png/augmented-confidence/outer50'),
        # ("NIST302a-M-confidence-augmented", 'data/NIST302/auxiliary/flat/M/500/plain/png/augmented-confidence/regular'),
        # ("NIST302a-M-inner50", 'data/NIST302/auxiliary/flat/M/500/plain/png/inner50'),
        # ("NIST302a-M-outer50", 'data/NIST302/auxiliary/flat/M/500/plain/png/outer50'),
        # ("NIST302a-M-inner60", 'data/NIST302/auxiliary/flat/M/500/plain/png/inner60'),
        # ("NIST302a-M-4-split-18to28", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/18to28'),
        # ("NIST302a-M-4-split-28to38", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/28to38'),
        # ("NIST302a-M-4-split-38to48", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/38to48'),
        # ("NIST302a-M-4-split-48to58", 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split/48to58'),
        # ("NIST4", 'data/sd04/png_txt/figs'),
        ("NIST4-easiest-MOC", 'data/sd04/png_txt/augmented-confidence-easiest/figs/'),
        ("NIST4-M-best-95-flip-rate-train-only", 'data/sd04/png_txt/best-95-flip-rate-train-only/figs/'),
        # ("NIST4-worse-flip", 'data/sd04/png_txt/worse-95-flip-rate/figs'),
        # ("NIST4-clean", 'data/sd04/png_txt/clean_lab/figs'),
        # ("NIST4-inner50-clean", 'data/sd04/png_txt/clean_lab/inner50'),
        # ("NIST4-outer50-clean", 'data/sd04/png_txt/clean_lab/outer50'),
        # ("NIST4-inner50-confidence", 'data/sd04/png_txt/augmented-confidence/inner50'),
        # ("NIST4-outer50-confidence", 'data/sd04/png_txt/augmented-confidence/outer50'),
        # ("NIST4-confidence-augmented", 'data/sd04/png_txt/augmented-confidence/figs'),
        # ("NIST4-inner50", 'data/sd04/png_txt/inner50'),
        # ("NIST4-outer50", 'data/sd04/png_txt/outer50'),
        # ("NIST4-inner60", 'data/sd04/png_txt/inner60')
        # ("SOCOfing", 'data/SOCOFing/Real'),
        # ("SOCOfing-confidence-augmented", 'data/SOCOFing/augmented-confidence/Real'),
        # ("SOCOFing-inner50", 'data/SOCOFing/inner50'),
        # ("SOCOFing-outer50", 'data/SOCOFing/outer50'),
        # ("SOCOFing-inner60", 'data/SOCOFing/inner60'),
    ]
