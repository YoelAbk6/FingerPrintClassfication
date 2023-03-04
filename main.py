from collections import defaultdict
from utils.data_loaders.data_loader import CustomImageDataset
from utils.models.lists_generator import *
from networks.train import *
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sn
import itertools
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


first_epochs = 1
full_train_epochs = 3
num_classes = 2
classes = ['Female', 'Male']
best_comb_occurences = defaultdict(int)
best_comb_percentage = defaultdict(float)


def save_conf_matrix(path, y_real, y_pred):
    cf_matrix = confusion_matrix(y_real, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path)
    plt.clf()


def save_performance_graph(train_data, test_data, num_epochs, title, save_path):
    epochs = np.linspace(1, num_epochs, num_epochs).astype(int)
    plt.plot(epochs, train_data, label='train')
    plt.plot(epochs, test_data, label='test')
    plt.title(title + ' graph')
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_sets = get_data_sets_list()

    for DS_name, DS_path in data_sets:
        print('=================================================================================================')
        print(f'Starting resarch best model for {DS_name}')
        print('=================================================================================================')

        data = CustomImageDataset(DS_path, device)
        train_dataloader, test_dataloader = data.get_train_and_test_data()

        # Generate lists
        models = get_models_list()
        losses = get_losses_list()
        optimizers = get_optimizers_list()
        learning_rates = get_learning_rates_list()
        best_model = best_loss = best_optimizer = best_lr = None
        best_model_state_dict = {}
        best_accuracy = 0
        best_iter = None
        best_y_pred, best_y_real = [], []
        best_accuracy_list_train, best_loss_list_train, best_accuracy_list_test, best_loss_list_test = [], [], [], []

        # Try all combinations to find the best one
        for iter in itertools.product(models, losses, optimizers, learning_rates):

            model_tuple, loss_tuple, optimizer_tuple, lr_tuple = iter
            model_name, model = model_tuple
            loss_name, loss = loss_tuple
            optimizer_name, optimizer = optimizer_tuple
            lr_name, lr = lr_tuple
            y_pred, y_real = [], []
            curr_accuracy_list_train, curr_loss_list_train, curr_accuracy_list_test, curr_loss_list_test = [], [], [], []
            print('=================================================================================================')
            print(
                f'Running with Model: {model_name}, Loss: {loss_name}, Optimizer: {optimizer_name}, Learning Rate: {lr_name}')
            print('=================================================================================================')

            curr_model = init_model(model, model_name, device, num_classes)

            for t in range(first_epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                accuracy_train, loss_train = train_loop(train_dataloader,
                                                        curr_model,
                                                        loss,
                                                        init_optimizer(optimizer, optimizer_name,
                                                                       curr_model, lr),
                                                        device)
                curr_accuracy_list_train.append(accuracy_train)
                curr_loss_list_train.append(loss_train)
                curr_accuracy, curr_y_pred, curr_y_real, loss_test = test_loop(
                    test_dataloader, curr_model, loss, device)
                accuracy_list_test.append(curr_accuracy)
                loss_list_test.append(loss_test)
                y_pred.extend(curr_y_pred)
                y_real.extend(curr_y_real)

            if len(best_accuracy_list_test) == 0 or max(curr_accuracy_list_test) > max(best_accuracy_list_test):
                best_accuracy_list_test = curr_accuracy_list_test
                best_accuracy_list_train = curr_accuracy_list_train
                best_loss_list_test = curr_loss_list_test
                best_loss_list_train = curr_loss_list_train
                best_y_pred = y_pred
                best_y_real = y_real

                best_accuracy = max(curr_accuracy_list_test)
                best_model = (model_name, curr_model)
                best_loss = loss_tuple
                best_optimizer = optimizer_tuple
                best_lr = lr_tuple
                best_iter = iter

        out_dir = f'./out/{DS_name}/research_best_model/{best_model[0]}'
        os.makedirs(out_dir, exist_ok=True)
        save_conf_matrix(
            f'{out_dir}/Confusion_Matrix.png', best_y_real, best_y_pred)

        save_performance_graph(best_accuracy_list_train, best_accuracy_list_test,
                               first_epochs, "Accuracy", f'{out_dir}/Accuracy_graph.png')
        save_performance_graph(best_loss_list_train, best_loss_list_test, first_epochs,
                               "Loss", f'{out_dir}/Loss_graph.png')
        print('=================================================================================================')
        print(f'Resarch best model is done for {DS_name}!')
        print(
            f'The best model and hyperparameters for {DS_name} are - Model: {best_model[0]}, Loss: {best_loss[0]}, Optimizer: {best_optimizer[0]}, Learning Rate: {best_lr[0]}')
        print('=================================================================================================')

        best_comb_occurences[best_iter] += 1
        if best_iter not in best_comb_percentage or best_comb_percentage[best_iter] < best_accuracy:
            best_comb_percentage[best_iter] = best_accuracy

    # Choose the model that performed the best across all datasets
    best_comb_iter = None
    most_used = -1
    best_accuracy = -1

    for iter in best_comb_occurences:
        if best_comb_occurences[iter] > most_used or (best_comb_occurences[iter] == most_used and best_comb_percentage[iter] > best_accuracy):
            most_used = best_comb_occurences[iter]
            best_accuracy = best_comb_percentage[iter]
            best_comb_iter = iter

    model_tuple, loss_tuple, optimizer_tuple, lr_tuple = best_comb_iter
    print('=================================================================================================')
    print(
        f'The best model and hyperparameters acorss all datasets are Model: {model_tuple[0]}, Loss: {loss_tuple[0]}, Optimizer: {optimizer_tuple[0]}, Learning Rate: {lr_tuple[0]}')
    print('=================================================================================================')

    print('=================================================================================================')
    print(f'Starting full training on each dataset with {model_tuple[0]}')
    print('=================================================================================================')

    for DS_name, DS_path in data_sets:

        data = CustomImageDataset(DS_path, device)
        train_dataloader, test_dataloader = data.get_train_and_test_data()
        accuracy_list_train, loss_list_train, accuracy_list_test, loss_list_test = [], [], [], []
        y_pred, y_real = [], []
        best_accuracy = 0
        for t in range(full_train_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            accuracy_train, loss_train = train_loop(train_dataloader,
                                                    best_model[1],
                                                    best_loss[1],
                                                    init_optimizer(best_optimizer[1], best_optimizer[0],
                                                                   best_model[1], best_lr[1]),
                                                    device)
            curr_accuracy, curr_y_pred, curr_y_real, loss_test = test_loop(
                test_dataloader, best_model[1], best_loss[1], device)
            accuracy_list_test.append(curr_accuracy)
            loss_list_test.append(loss_test)
            y_pred.extend(curr_y_pred)
            y_real.extend(curr_y_real)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy

        # best_model_state_dict.append({DS_name, best_model[1].state_dict()})

        out_dir = f'./out/{DS_name}/best_model_performance/{best_model[0]}/'
        os.makedirs(out_dir, exist_ok=True)
        save_conf_matrix(
            f'{out_dir}/Confusion_Matrix.png', y_real, y_pred)

        save_performance_graph(accuracy_list_train, accuracy_list_test,
                               first_epochs, "Accuracy", f'{out_dir}/Accuracy_graph.png')
        save_performance_graph(loss_list_train, loss_list_test, first_epochs,
                               "Loss", f'{out_dir}/Loss_graph.png')
        print('=================================================================================================')
        print(
            f'Train best model on {DS_name} is done, Best accuracy reached: {best_accuracy:>0.2f}%, model is saved in best_model_state_dict')
        print('=================================================================================================')

    print("The run is done")


if __name__ == '__main__':
    main()
