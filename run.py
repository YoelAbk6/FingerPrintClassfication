from collections import defaultdict
from utils.data_loaders.data_loader import CustomImageDataset
from utils.models.lists_generator import *
from utils.output_generator import *
from networks.train import *
from evaluate import *
from utils.arguments_parser import ArgumentParser
import os
import random
from utils import definitions

torch.manual_seed(definitions.RANDOM_SEED)
np.random.seed(definitions.RANDOM_SEED)
random.seed(definitions.RANDOM_SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models = {
    'Resnet50': models.resnet50,
    'Resnet101': models.resnet101,
    'VGG-19': models.vgg19,
    'Mobilenet-v2': models.mobilenet_v2
}

losses = {
    'CrossEntropyLoss': nn.CrossEntropyLoss(),
}

optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'Adagrad': optim.Adagrad
}


def parse_args():
    """ Arguments parsing

    Returns:
        argparse.Namespace -- the arguments
     """

    parser = ArgumentParser()
    parser.add_argument("--model", "-m", required=True,
                        help="Model name")
    parser.add_argument("--optimizer", "-opt", required=True,
                        help="Optimizer name")
    parser.add_argument("--learningRate", "-lr", required=True,
                        help="Learning rate")
    parser.add_argument("--loss", "-l", required=True,
                        help="Loss function name")
    try:
        args = parser.parse_args()
        return args
    except SystemExit as exc:
        exit(exc.code)


def main():

    args = parse_args()
    num_epochs = 15
    num_classes = 2

    data_sets = get_data_sets_list()

    model_name = args.model
    model = models.get(model_name)
    optimizer_name = args.optimizer
    optimizer = optimizers.get(optimizer_name)
    lr = float(args.learningRate)
    loss = losses.get(args.loss)

    # Loop through all datasets best model
    for DS_name, DS_path in data_sets:
        print('=================================================================================================')
        print(
            f'Training {model_name} on {DS_name} started')
        print('=================================================================================================')

        data = CustomImageDataset(DS_path, device, num_classes)
        train_dataloader, test_dataloader = data.get_train_and_test_data()
        accuracy_list_train, loss_list_train, accuracy_list_test, loss_list_test = [], [], [], []
        y_pred_list, y_real_list = [], []
        best_accuracy = 0

        curr_model = init_model(model, model_name, device, num_classes)
        # Train and test loop
        for t in range(num_epochs):
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

        # Handle outputs
        out_dir = f'./out/{DS_name}/simple_run_rs={definitions.RANDOM_SEED}/{model_name}/'
        os.makedirs(out_dir, exist_ok=True)
        save_conf_matrix(
            f'{out_dir}/Confusion_Matrix.png', y_real_list, y_pred_list)

        save_classification_report(y_real_list, y_pred_list, f'{out_dir}/classification_report.txt')

        save_performance_graph(accuracy_list_train, accuracy_list_test,
                               num_epochs, "Accuracy", f'{out_dir}/Accuracy_graph.png')
        save_performance_graph(loss_list_train, loss_list_test, num_epochs,
                               "Loss", f'{out_dir}/Loss_graph.png')
        # Save model
        torch.save(curr_model.state_dict(), f'{out_dir}/my_model.pt')
        save_test_data(test_dataloader, out_dir)
        print('=================================================================================================')
        print(
            f'Train {model_name} on {DS_name} is done, best accuracy reached: {best_accuracy:>0.2f}%, model is saved in best_model_state_dict')
        print('=================================================================================================')

    print("The run is done")


if __name__ == '__main__':
    main()
