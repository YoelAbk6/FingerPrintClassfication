from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

classes = ['Female', 'Male']


def save_test_data(dataloader, out_dir):
    f = open(os.path.join(out_dir, "testList.txt"), "w")
    with torch.no_grad():
        for X, y, paths in dataloader:
            for path in paths:
                f.write(path + '\n')


def save_conf_matrix(path, y_real, y_pred):
    cf_matrix = confusion_matrix(y_real, y_pred, normalize='true')
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


def copy_pictures_from_path_to_location(pathes, new_folder_name):
    for path in pathes:
        # get the directory, filename, and extension
        dir_path, file_name = os.path.split(path)
        file_name, ext = os.path.splitext(file_name)

        # get the parent directory of the current directory
        parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

        # construct the new directory path
        new_dir_path = os.path.join(
            parent_dir_path, new_folder_name, os.path.basename(dir_path))

        # create the new directory if it doesn't exist
        os.makedirs(new_dir_path, exist_ok=True)

        # construct the new file path
        new_file_path = os.path.join(new_dir_path, file_name + ext)

        # copy the file to the new location
        shutil.copy(path, new_file_path)


def save_classification_report(y_test, y_pred, save_path):

    report = classification_report(y_test, y_pred, target_names=classes)

    with open(save_path, 'w') as f:
        f.write(report)


def print_and_save(message, path):

    print(message)

    with open(f'{path}/log.txt', 'a+') as f:
        f.write(f'{message}\n')
