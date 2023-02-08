import os
import torch
device = torch.device("cuda:0")
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, image_dir):
        # super(ImageDataset, self).__init__()
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image_dir = image_dir

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_name = os.listdir(self.image_dir)[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.transform(image)
        label = self.get_label_from_filename(image_name)
        return image.to(device), label

    def get_label_from_filename(self, image_name):
        file_name = image_name.replace('.png', '')
        return 1 if file_name.split('_')[-2] == 'M' else 0

    def get_train_and_test_data(self):
        train_size = int(0.8 * self.__len__())
        test_size = self.__len__() - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            self, [train_size, test_size])

        train_dataset = train_dataset
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True), torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=True)
