import math
import torch
import numpy as np
from PIL import Image
from networks.train import *
import matplotlib.pyplot as plt
from utils.output_generator import *
from utils.models.lists_generator import *
from utils.data_loaders.data_loader import CustomImageDataset
from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues
import random
from utils import definitions

torch.manual_seed(definitions.RANDOM_SEED)
np.random.seed(definitions.RANDOM_SEED)
random.seed(definitions.RANDOM_SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
M = 'M'
F = 'F'
num_classes = 2
TOP_ISSUES = 45


def load_model(model_path, VGG_19=True):
    if VGG_19:
        input_model = 'VGG-19'
    else:
        input_model = model_path.split("/")[-2].replace("-augmented", "")

    models = get_models_list()
    for model_name, model in models:
        if input_model == model_name:
            inited_model = init_model(model, model_name, device, 2)
            inited_model.load_state_dict(torch.load(model_path))
            return inited_model
    return None


def show_side_by_side(images: list, ncols: int = 6) -> None:
    f, axes = plt.subplots(
        nrows=math.ceil(len(images) / ncols), ncols=ncols, sharex=True, sharey=True)

    # Loop through the images and plot them in the subplots
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img, title = images[i]
            ax.imshow(img)
            ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def imshow(img):
    npimg = img.cpu().numpy()
    return np.transpose(npimg, (1, 2, 0))


def plot_images(dataset, show_labels=False):
    plt.rcParams["figure.figsize"] = (9, 7)
    for i in range(TOP_ISSUES):
        X, y = dataset[i][0:2]
        print(dataset[i][2])
        ax = plt.subplot(9, 5, i + 1)
        if show_labels:
            ax.set_title(txt_classes[int(y)])
        ax.imshow(imshow(X))
        ax.axis('off')
    plt.show()


txt_classes = {0: 'female',
               1: 'male'}


def visualize_outliers(idxs, data):
    data_subset = torch.utils.data.Subset(data, idxs)
    plot_images(data_subset, True)


def predict(model, DS_path):

    data = CustomImageDataset(DS_path, device, num_classes, use_file=True)
    test_dataloader = data.get_data()
    loss = nn.CrossEntropyLoss()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for X, y, paths in test_dataloader:
            y_true.extend(y.data.numpy())
            y = y.to(device)
            pred = model(X)
            y_pred.extend(torch.argmax(pred, 1).data.cpu().numpy())
            test_loss += loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # print_and_save(
    #     f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n", path)

    # with torch.no_grad():
    #     for X, y, paths in test_dataloader:
    #         y = y.to(device)
    #         pred = model(X)
    #         imgs = []
    #         for path, label, predicted in zip(paths, y, pred.argmax(1) == y):
    #             if len(imgs) < 16:
    #                 title = f'{M if label == 1 else F} - {str(predicted.item())}'
    #                 imgs.append(
    #                     (np.array(Image.open(path).convert('RGB')), title))
    #         show_side_by_side(imgs)


def clean_lab(model, train_DS_path, output_path='', plot_dist=False, plot_top=False):
    ood = OutOfDistribution()

    data = CustomImageDataset(train_DS_path, device, num_classes, use_file=True)
    # dataloader = data.get_data()
    train_dataloader = data.get_train_data()
    pred_probs = []
    labels = []

    with torch.no_grad():
        for X, y, paths in train_dataloader:
            y = y.cpu()
            X = X.cpu()
            pred = torch.softmax(model(X), dim=1)
            pred_probs.append(pred.cpu().numpy())
            labels.append(y.numpy())

    pred_probs = np.concatenate(pred_probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    ood_features_scores = ood.fit_score(pred_probs=pred_probs, labels=labels)

    # Visualize top issues
    if plot_top:
        top_ood_features_idxs = find_top_issues(
            quality_scores=ood_features_scores, top=TOP_ISSUES)
        visualize_outliers(top_ood_features_idxs, data)

    # 5th percentile of the train_data distribution
    fifth_percentile = np.percentile(ood_features_scores, 5)

    # Plot outlier_score distribution and the 5th percentile cutoff
    if plot_dist:
        plt.figure(figsize=(8, 5))
        plt_range = [min(ood_features_scores), max(ood_features_scores)]
        plt.hist(ood_features_scores, range=plt_range, bins=50)
        plt.title('ood_features_scores distribution')
        plt.xlabel('Outlier score')
        plt.ylabel('Frequency')
        plt.axvline(x=fifth_percentile, color='red', linewidth=2)
        plt.savefig(f'{output_path}ood.png')

    sorted_idxs = ood_features_scores.argsort()
    ood_features_scores = ood_features_scores[sorted_idxs]
    ood_features_indices = sorted_idxs[ood_features_scores < fifth_percentile]
    clean_data = [data.image_paths[i] for i in range(
        len(data.image_paths)) if i not in ood_features_indices]

    # for i in range(len(test_dataloader.dataset.indices)):
    #     clean_data.append(data.image_paths[test_dataloader.dataset.indices[i]])

    print(f'{len(data.image_paths) - len(clean_data)} pictures will be removed from the DB, with conf_threshold = {fifth_percentile}')
    copy_pictures_from_path_to_location(clean_data, 'clean_lab-final')


def filter_images_by_confidence_score(model, train_DS_path, test_DS_path, output_path="", plot=False, create_DB=True):
    train_data = CustomImageDataset(train_DS_path, device, num_classes, use_file=True)
    test_data = CustomImageDataset(test_DS_path, device, num_classes, use_file=True)
    train_dataloader, test_dataloader = train_data.get_train_data(), test_data.get_data()

    confidences = []
    with torch.no_grad():
        for i, (X, y, paths) in enumerate(train_dataloader):
            probs = torch.softmax(model(X), dim=1)
            confidence = torch.max(probs, dim=1)[
                0] - torch.min(probs, dim=1)[0]

            confidences.extend(confidence.cpu().numpy())

    fifth_percentile = np.percentile(confidences, 95)
    confidences = np.array(confidences, dtype=np.float32)
    sorted_idxs = confidences.argsort()
    confidences = confidences[sorted_idxs]
    low_conf = sorted_idxs[confidences < fifth_percentile]

    data_to_save = [train_data.image_paths[low_conf[i]]
                    for i in range(len(low_conf))]

    # for i in range(len(test_data.image_paths)):
    #     data_to_save.append(test_data.image_paths[i])

    print(f'{len(train_data.image_paths) - len(low_conf)} pictures will be removed from the DB, with conf_threshold = {fifth_percentile}')

    if plot:
        plt.hist(confidences, bins=50)
        plt.axvline(x=fifth_percentile, color='red', linestyle='--')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Distribution of Confidence Scores')
        plt.show()
        plt.savefig(f'{output_path}/conf_dist.png')

    if create_DB:
        copy_pictures_from_path_to_location(data_to_save, 'augmented-confidence-easiest')
