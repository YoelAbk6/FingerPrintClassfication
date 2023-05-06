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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models = {
    'Resnet50': models.resnet50,
    'Resnet18': models.resnet18,
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
    num_epochs = 20
    num_classes = 2

    data_sets = get_data_sets_list()

    model_name = args.model
    model = models.get(model_name)
    optimizer_name = args.optimizer
    optimizer = optimizers.get(optimizer_name)
    lr = float(args.learningRate)
    loss = losses.get(args.loss)
    y_pred = y_real = None

    # Loop through all datasets best model
    for DS_name, DS_path in data_sets:

        out_dir = f'./out/{DS_name}/simple_run_rs={definitions.RANDOM_SEED}/{model_name}/'
        os.makedirs(out_dir, exist_ok=True)

        print_and_save(
            '=================================================================================================', out_dir)
        print_and_save(
            f'Training {model_name} on {DS_name} started', out_dir)
        print_and_save(
            '=================================================================================================', out_dir)

        data = CustomImageDataset(DS_path, device, num_classes)
        train_dataloader, test_dataloader = data.get_train_and_test_data()
        accuracy_list_train, loss_list_train, accuracy_list_test, loss_list_test = [], [], [], []
        best_test_accuracy = best_train_accuracy = 0
        best_y_pred, best_y_real = [], []

        curr_model = init_model(model, model_name, device, num_classes)
        # Train and test loop
        for t in range(num_epochs):
            print_and_save(f"Epoch {t+1}\n-------------------------------", out_dir)
            curr_optim = init_optimizer(
                optimizer, optimizer_name, curr_model, lr)

            accuracy_train, loss_train = train_loop(train_dataloader,
                                                    curr_model,
                                                    loss,
                                                    curr_optim,
                                                    device,
                                                    out_dir)
            accuracy_list_train.append(accuracy_train)
            loss_list_train.append(loss_train)

            curr_accuracy, y_pred, y_real, loss_test = test_loop(
                test_dataloader,
                curr_model,
                loss,
                device,
                out_dir)
            accuracy_list_test.append(curr_accuracy)
            loss_list_test.append(loss_test)

            if curr_accuracy > best_test_accuracy:
                best_test_accuracy = curr_accuracy
                best_y_real = y_real
                best_y_pred = y_pred
                # Save model
                torch.save(curr_model.state_dict(), f'{out_dir}/my_model.pt')

            best_train_accuracy = max(best_train_accuracy, accuracy_train)

        # Handle outputs
        save_conf_matrix(
            f'{out_dir}/Confusion_Matrix.png', best_y_real, best_y_pred)

        save_classification_report(best_y_real, best_y_pred, f'{out_dir}/classification_report.txt')

        save_performance_graph(accuracy_list_train, accuracy_list_test,
                               num_epochs, "Accuracy", f'{out_dir}/Accuracy_graph.png')
        save_performance_graph(loss_list_train, loss_list_test, num_epochs,
                               "Loss", f'{out_dir}/Loss_graph.png')
        save_test_data(test_dataloader, out_dir)
        print_and_save(
            '=================================================================================================', out_dir)
        print_and_save(
            f'Train {model_name} on {DS_name} is done\nBest accuracy train reached: {best_train_accuracy:>0.2f}%\nBest accuracy test reached: {best_test_accuracy:>0.2f}%\nModel is saved in best_model_state_dict', out_dir)
        print_and_save(
            '=================================================================================================', out_dir)

    print("The run is done")


if __name__ == '__main__':
    main()
