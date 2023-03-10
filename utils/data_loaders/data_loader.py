import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, image_source, device, use_file=False):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.device = device
        self.use_file = use_file

        if self.use_file:
            self.image_paths = []
            with open(image_source, 'r') as f:
                for line in f:
                    self.image_paths.append(line.strip())
        else:
            self.image_paths = [os.path.join(image_source, filename) for filename in os.listdir(image_source)]

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

        train_dataset = train_dataset
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True), torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=True)

    def get_data(self):
        return torch.utils.data.DataLoader(self, batch_size=32, shuffle=True)
    
