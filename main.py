import itertools
from networks.train import *
from utils.models.lists_generator import *
from utils.data_loaders.data_loader import CustomImageDataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
first_epochs = 5
full_train_epochs = 30
num_classes = 2


def main():

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    # Load data
    # path = 'data\\NIST302\\images\\auxiliary\\flat\\M\\500\\plain\\png'
    path = '/home/uoriko/FingerPrintClassfication/data/equal'
    # path = 'D:\\data\\NIST302\\images\\auxiliary\\flat\\M\\500\\plain\\png\\equal'
    data = CustomImageDataset(path, device)
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
        print('=================================================================================================')
        print(
            f'Start training with - Model: {model_name}, Loss: {loss_name}, Optimizer: {optimizer_name}, Learning Rate: {lr_name}')
        print('=================================================================================================')

        curr_model = init_model(model, model_name, device, num_classes)

        # model = nn.DataParallel(model)

        # if hasattr(model.module, 'classifier'):
        #     num_features = model.module.classifier[-1].in_features
        #     model.module.classifier[-1] = nn.Linear(num_features, num_classes)
        # else:
        #     num_features = model.module.fc.in_features
        #     model.module.fc = nn.Linear(num_features, num_classes)

        # model.to(device)
        for t in range(first_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader,
                       curr_model,
                       loss,
                       init_optimizer(optimizer, optimizer_name, curr_model, lr),
                       device)
            test_loop(test_dataloader, curr_model, loss, device)
        print("Done!")


if __name__ == '__main__':
    main()
