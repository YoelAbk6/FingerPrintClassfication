import math
import torch
import numpy as np
from PIL import Image
from networks.train import *
import matplotlib.pyplot as plt
from utils.output_generator import *
from utils.models.lists_generator import *
from utils.data_loaders.data_loader import CustomImageDataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
M = 'M'
F = 'F'

def load_model(model_path):
    models = get_models_list()
    input_model = model_path.split("/")[-2]
    for model_name, model in models:
        if input_model == model_name:
            inited_model = init_model(model, model_name, device, 2)
            inited_model.load_state_dict(torch.load(model_path))
            return inited_model
    return None

def show_side_by_side(images: list, ncols: int = 6) -> None:
    f, axes = plt.subplots(
        nrows=math.ceil(len(images)/ncols), ncols=ncols, sharex=True, sharey=True)

    # Loop through the images and plot them in the subplots
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img, title = images[i]
            ax.imshow(img)
            ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def predict(model, DS_path):

    data = CustomImageDataset(DS_path, device, use_file=True)
    test_dataloader = data.get_data()

    with torch.no_grad():
        for X, y, paths in test_dataloader:
            y = y.to(device)
            pred = model(X)
            imgs = []
            for path, label, predicted in zip(paths, y, pred.argmax(1) == y):
                if len(imgs) < 16:
                    title = f'{M if label == 1 else F} - {str(predicted.item())}'
                    imgs.append((np.array(Image.open(path).convert('RGB')), title))
            show_side_by_side(imgs)
