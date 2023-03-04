
import torch
from networks.train import *
from utils.models.lists_generator import *
from utils.data_loaders.data_loader import CustomImageDataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    models = get_models_list()
    input_model = model_path.split("\\")[-2]
    for model_name, model in models:
        if input_model == model_name:
            inited_model = init_model(model, model_name, device, 2)
            inited_model.load_state_dict(torch.load(model_path))
            return inited_model
    return None


def predict(model, DS_path):

    data = CustomImageDataset(DS_path, device)
    train_dataloader, test_dataloader = data.get_train_and_test_data()

    # model.eval()
    # output = model(train_dataloader)
    with torch.no_grad():
        for X, y in train_dataloader:
            y = y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            pass
