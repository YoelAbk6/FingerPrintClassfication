import numpy as np
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import os
import random
import random

torch.manual_seed(1997)
np.random.seed(1997)
random.seed(1997)

BATCH_SIZE = 32


class CustomImageDataset(Dataset):
    def __init__(self, image_source, device, num_classes, use_file=False):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.device = device
        self.use_file = use_file
        self.classes = num_classes

        if self.use_file:
            self.image_paths = []
            with open(image_source, 'r') as f:
                for line in f:
                    self.image_paths.append(line.strip())
        else:
            self.image_paths = [os.path.join(
                image_source, filename) for filename in os.listdir(image_source)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.transform(image)
        label = self.get_label_from_filename(image_path)
        return image.to(self.device), label, image_path

    def get_label_from_filename(self, image_path):
        image_name = os.path.basename(image_path)
        return 1 if image_name.split('_')[-2] == 'M' else 0

    def get_train_and_test_data(self):
        train_size = int(0.8 * self.__len__())
        test_size = self.__len__() - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            self, [train_size, test_size])

        # Get the class distribution of the train dataset
        class_distribution = [0] * self.classes
        for _, label, _ in train_dataset:
            class_distribution[label] += 1

        # Compute the weight of each sample in the train dataset
        weights = [1.0 / class_distribution[label]
                   for _, label, _ in train_dataset]

        # Create a sampler using the computed weights
        sampler = WeightedRandomSampler(weights, len(weights))

        return torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler), torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def get_data(self):
        return torch.utils.data.DataLoader(self, batch_size=BATCH_SIZE, shuffle=False)
