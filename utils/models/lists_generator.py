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


# "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"
# "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"
# "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"

# name, train_path(folder), test_path(list)
def get_data_sets_list():
    return [
        # ("NIST302a-M",
        #  "/home/uoriko/FingerPrintClassfication/data/train/NIST302_TrainData.list",
        #  "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),
        # ("NIST4",
        #  "/home/uoriko/FingerPrintClassfication/data/train/NIST4_TrainData.list",
        #  "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),
        # ("SOCOfing",
        #  "/home/uoriko/FingerPrintClassfication/data/train/SOCOfing_TrainData.list",
        #  "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),
        ("NIST302a-M-MOC-hard-final",
         "/home/uoriko/FingerPrintClassfication/data/NIST302/auxiliary/flat/M/500/plain/png/MOC-hard-data-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),

        ("NIST302a-M-MOC-easy-final",
         "/home/uoriko/FingerPrintClassfication/data/NIST302/auxiliary/flat/M/500/plain/png/MOC-easy-data-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),

        ("NIST302a-M-clean-lab-final",
         "/home/uoriko/FingerPrintClassfication/data/NIST302/auxiliary/flat/M/500/plain/png/clean_lab-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),

        ("NIST302a-M-flip-easy-final",
         "/home/uoriko/FingerPrintClassfication/data/NIST302/auxiliary/flat/M/500/plain/png/flip-easy-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),

        ("NIST302a-M-flip-hard-final",
         "/home/uoriko/FingerPrintClassfication/data/NIST302/auxiliary/flat/M/500/plain/png/flip-hard-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST302_TestData.list"),


        ("NIST4-MOC-hard-final",
         "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/MOC-hard-data-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),

        ("NIST4-MOC-easy-final",
         "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/MOC-easy-data-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),

        ("NIST4-clean-lab-final",
         "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/clean_lab-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),

        ("NIST4-flip-easy-final",
         "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/flip-easy-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),

        ("NIST4-flip-hard-final",
         "/home/uoriko/FingerPrintClassfication/data/sd04/png_txt/flip-hard-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/NIST4_TestData.list"),


        ("SOCOfing-MOC-hard-final",
         "/home/uoriko/FingerPrintClassfication/data/SOCOFing/MOC-hard-data-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),

        ("SOCOfing-MOC-easy-final",
         "/home/uoriko/FingerPrintClassfication/data/SOCOFing/MOC-easy-data-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),

        ("SOCOfing-clean-lab-final",
         "/home/uoriko/FingerPrintClassfication/data/SOCOFing/clean_lab-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),

        ("SOCOfing-flip-easy-final",
         "/home/uoriko/FingerPrintClassfication/data/SOCOFing/flip-easy-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),

        ("SOCOfing-flip-hard-final",
         "/home/uoriko/FingerPrintClassfication/data/SOCOFing/flip-hard-final/",
         "/home/uoriko/FingerPrintClassfication/data/test/SOCOfing_TestData.list"),
    ]
