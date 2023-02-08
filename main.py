import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils.data_loaders.data_loader import CustomImageDataset
from utils.models.lists_generator import *
from networks.train import *
import itertools
first_epochs = 5
full_train_epochs = 30


def main():
    # Load data
    path = 'data\\NIST302\\images\\auxiliary\\flat\\M\\500\\plain\\png\\equal'
    path = '/home/uoriko/FingerPrintClassfication/data/equal'
    device = torch.device("cuda:0")
    data = CustomImageDataset(path)
    train_dataloader, test_dataloader = data.get_train_and_test_data()

    # Generate lists
    models = get_models_list()
    losses = get_losses_list()
    optimizers = get_optimizers_list()
    learning_rates = get_learning_rates_list()
    best_model = None

    # Try all combinations to find the best one
    for iter in itertools.product(models, losses, optimizers, learning_rates):

        model_tuple, loss_tuple, optimizer_tuple, lr_tuple = iter
        model_name, model = model_tuple
        loss_name, loss = loss_tuple
        optimizer_name, optimizer = optimizer_tuple
        lr_name, lr = lr_tuple
        print(
            f'Start training with - Model: {model_name}, Loss: {loss_name}, Optimizer: {optimizer_name}, Learning Rate: {lr_name}')

        model = nn.DataParallel(model)
        model.to(device)
        for t in range(first_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss, optimizer(
                model.module.fc.parameters(), lr=lr, momentum=0.9))
            test_loop(test_dataloader, model, loss)
        print("Done!")


if __name__ == '__main__':
    main()
