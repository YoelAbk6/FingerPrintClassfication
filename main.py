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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    # path = 'data\\NIST302\\images\\auxiliary\\flat\\M\\500\\plain\\png'
    path = '/home/uoriko/FingerPrintClassfication/data/equal'
    path = 'D:\\data\\NIST302\\images\\auxiliary\\flat\\M\\500\\plain\\png\\equal'
    data = CustomImageDataset(path, device)
    train_dataloader, test_dataloader = data.get_train_and_test_data()

    # Generate lists
    models = get_models_list()
    losses = get_losses_list()
    optimizers = get_optimizers_list()
    learning_rates = get_learning_rates_list()
    best_model = best_loss = best_optimizer = best_lr = None
    best_model_state_dict = None
    best_accuracy = 0

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

        for t in range(first_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader,
                       curr_model,
                       loss,
                       init_optimizer(optimizer, optimizer_name,
                                      curr_model, lr),
                       device)
            curr_accuracy = test_loop(
                test_dataloader, curr_model, loss, device)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_model = (model_name, curr_model)
                best_loss = loss_tuple
                best_optimizer = optimizer_tuple
                best_lr = lr_tuple

    print('=================================================================================================')
    print("Resarch best model is done!")
    print(
        f'The best model and hyperparameters are - Model: {best_model[0]} Loss: {best_loss[0]}, Optimizer: {best_optimizer[0]}, Learning Rate: {best_lr[0]}')
    print('=================================================================================================')

    print('=================================================================================================')
    print('Starting with full training')
    print('=================================================================================================')
    for t in range(full_train_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader,
                   best_model[1],
                   best_loss[1],
                   init_optimizer(best_optimizer[1], best_optimizer[0],
                                  best_model[1], best_lr[1]),
                   device)
        curr_accuracy = test_loop(
            test_dataloader, best_model[1], best_loss[1], device)
        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            # best_model_state_dict = best_model[1].state_dict()

    # for param_tensor in best_model_state_dict:
    #     print(param_tensor, "\t", best_model_state_dict[param_tensor].size())

    print('=================================================================================================')
    print("Train best model is done!")
    print(
        f'Best accuracy reached: {best_accuracy}, model is saved in best_model_state_dict')
    ('=================================================================================================')


if __name__ == '__main__':
    main()
