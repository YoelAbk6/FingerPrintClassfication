from collections import defaultdict
from utils.data_loaders.data_loader import CustomImageDataset
from utils.models.lists_generator import *
from utils.output_generator import *
from networks.train import *
from evaluate import *
import itertools
import os
import random

random.seed(1997)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():

    search_best_model_epochs = 1
    full_train_epochs = 5
    num_classes = 2
    best_comb_occurences = defaultdict(int)
    best_comb_percentage = defaultdict(float)

    data_sets = get_data_sets_list()

    # Loop through all datasets research
    for DS_name, DS_path in data_sets:
        print('=================================================================================================')
        print(f'Starting resarch best model for {DS_name}')
        print('=================================================================================================')

        data = CustomImageDataset(DS_path, device, num_classes)
        train_dataloader, test_dataloader = data.get_train_and_test_data()

        # Generate lists
        models = get_models_list()
        losses = get_losses_list()
        optimizers = get_optimizers_list()
        learning_rates = get_learning_rates_list()
        best_model = best_loss = best_optimizer = best_lr = None
        best_accuracy = 0
        best_iter = None
        best_y_pred, best_y_real = [], []
        best_accuracy_list_train, best_loss_list_train, best_accuracy_list_test, best_loss_list_test = [], [], [], []

        # Try all combinations to find the best one
        for iter in itertools.product(models, losses, optimizers, learning_rates):

            # Extract hyperparameters
            model_tuple, loss_tuple, optimizer_tuple, lr_tuple = iter
            model_name, model = model_tuple
            loss_name, loss = loss_tuple
            optimizer_name, optimizer = optimizer_tuple
            lr_name, lr = lr_tuple
            y_pred_list, y_real_list = [], []
            curr_accuracy_list_train, curr_loss_list_train, curr_accuracy_list_test, curr_loss_list_test = [], [], [], []
            print('=================================================================================================')
            print(
                f'Running with Model: {model_name}, Loss: {loss_name}, Optimizer: {optimizer_name}, Learning Rate: {lr_name}')
            print('=================================================================================================')

            # Init model to start the train
            curr_model = init_model(model, model_name, device, num_classes)

            # Train and test loop
            for t in range(search_best_model_epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                curr_optim = init_optimizer(
                    optimizer, optimizer_name, curr_model, lr)
                accuracy_train, loss_train = train_loop(train_dataloader,
                                                        curr_model,
                                                        loss,
                                                        curr_optim,
                                                        device)
                curr_accuracy_list_train.append(accuracy_train)
                curr_loss_list_train.append(loss_train)
                curr_accuracy, curr_y_pred, curr_y_real, loss_test = test_loop(
                    test_dataloader, curr_model, loss, device)
                curr_accuracy_list_test.append(curr_accuracy)
                curr_loss_list_test.append(loss_test)
                y_pred_list.extend(curr_y_pred)
                y_real_list.extend(curr_y_real)

            # Update the best model if needed
            if len(best_accuracy_list_test) == 0 or max(curr_accuracy_list_test) > max(best_accuracy_list_test):
                best_accuracy_list_test = curr_accuracy_list_test
                best_accuracy_list_train = curr_accuracy_list_train
                best_loss_list_test = curr_loss_list_test
                best_loss_list_train = curr_loss_list_train
                best_y_pred = y_pred_list
                best_y_real = y_real_list

                best_accuracy = max(curr_accuracy_list_test)
                best_iter = iter

        # Handle outputs research
        out_dir = f'./out/{DS_name}/research_best_model/{best_iter[0][0]}'
        os.makedirs(out_dir, exist_ok=True)
        save_conf_matrix(
            f'{out_dir}/Confusion_Matrix.png', best_y_real, best_y_pred)

        save_performance_graph(best_accuracy_list_train, best_accuracy_list_test,
                               search_best_model_epochs, "Accuracy", f'{out_dir}/Accuracy_graph.png')
        save_performance_graph(best_loss_list_train, best_loss_list_test, search_best_model_epochs,
                               "Loss", f'{out_dir}/Loss_graph.png')
        print('=================================================================================================')
        print(f'Resarch best model is done for {DS_name}!')
        print(
            f'The best model and hyperparameters for {DS_name} are - Model: {best_iter[0][0]}, Loss: {best_iter[1][0]}, Optimizer: {best_iter[2][0]}, Learning Rate: {best_iter[3][0]}')
        print('=================================================================================================')

        best_comb_occurences[best_iter] += 1
        if best_iter not in best_comb_percentage or best_comb_percentage[best_iter] < best_accuracy:
            best_comb_percentage[best_iter] = best_accuracy

    best_comb_iter = None
    most_used = -1
    best_accuracy = -1

    # Choose the model that performed the best across all datasets
    for iter in best_comb_occurences:
        if best_comb_occurences[iter] > most_used or (best_comb_occurences[iter] == most_used and best_comb_percentage[iter] > best_accuracy):
            most_used = best_comb_occurences[iter]
            best_accuracy = best_comb_percentage[iter]
            best_comb_iter = iter

    model_tuple, loss_tuple, optimizer_tuple, lr_tuple = best_comb_iter
    model_name, model = model_tuple
    loss_name, loss = loss_tuple
    optimizer_name, optimizer = optimizer_tuple
    lr_name, lr = lr_tuple
    print('=================================================================================================')
    print(
        f'The best model and hyperparameters acorss all datasets are Model: {model_tuple[0]}, Loss: {loss_tuple[0]}, Optimizer: {optimizer_tuple[0]}, Learning Rate: {lr_tuple[0]}')
    print('=================================================================================================')

    print('=================================================================================================')
    print(f'Starting full training on each dataset with {model_tuple[0]}')
    print('=================================================================================================')

    # Loop through all datasets best model
    for DS_name, DS_path in data_sets:

        data = CustomImageDataset(DS_path, device, num_classes)
        train_dataloader, test_dataloader = data.get_train_and_test_data()
        accuracy_list_train, loss_list_train, accuracy_list_test, loss_list_test = [], [], [], []
        y_pred_list, y_real_list = [], []
        best_accuracy = 0

        curr_model = init_model(model, model_name, device, num_classes)
        # Train and test loop
        for t in range(full_train_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            curr_optim = init_optimizer(
                optimizer, optimizer_name, curr_model, lr)
            accuracy_train, loss_train = train_loop(train_dataloader,
                                                    curr_model,
                                                    loss,
                                                    curr_optim,
                                                    device)
            accuracy_list_train.append(accuracy_train)
            loss_list_train.append(loss_train)
            curr_accuracy, curr_y_pred, curr_y_real, loss_test = test_loop(
                test_dataloader, curr_model, loss, device)
            accuracy_list_test.append(curr_accuracy)
            loss_list_test.append(loss_test)
            y_pred_list.extend(curr_y_pred)
            y_real_list.extend(curr_y_real)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy

        # Handle outputs best model
        out_dir = f'./out/{DS_name}/best_model_performance/{model_name}/'
        os.makedirs(out_dir, exist_ok=True)
        save_conf_matrix(
            f'{out_dir}/Confusion_Matrix.png', y_real_list, y_pred_list)

        save_performance_graph(accuracy_list_train, accuracy_list_test,
                               full_train_epochs, "Accuracy", f'{out_dir}/Accuracy_graph.png')
        save_performance_graph(loss_list_train, loss_list_test, full_train_epochs,
                               "Loss", f'{out_dir}/Loss_graph.png')
        # Save model
        torch.save(curr_model.state_dict(), f'{out_dir}/my_model.pt')
        save_test_data(test_dataloader, out_dir)
        print('=================================================================================================')
        print(
            f'Train best model on {DS_name} is done, best accuracy reached: {best_accuracy:>0.2f}%, model is saved in best_model_state_dict')
        print('=================================================================================================')

    print("The run is done")


def evaluate():

    for root, dirs, files in os.walk("./out/NIST4"):
        for file in files:
            if file.endswith('.pt'):
                model = load_model(os.path.join(root, file))
                model.eval()
                if model is not None:
                    #predict(
                     #   model, './out/NIST4/best_model_performance/VGG-19/testList.txt')
                    clean_lab(
                        model, './out/NIST4/best_model_performance/VGG-19/testList.txt')


if __name__ == '__main__':
    main()
    # evaluate()
