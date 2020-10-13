import argparse
import glob
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import Dataset


def save_plot(fig_name: str, line1: list, line2: list, line3: list, metric: str, n: int):

    plt.figure(n)
    plt.plot(np.array(line1), 'r', label='train')
    plt.plot(np.array(line2), 'b', label='test MSE')
    plt.plot(np.array(line3), 'g', label='test MAE')
    plt.title(metric)
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')

    plt.savefig(fig_name)

class Trainer(object):

    def __init__(self, model, train_name, variable):
        self.device = self.get_device()
        self.model = model
        self.train_name = train_name
        self.model = self.model.to(self.device)
        self.is_classification = (variable != 'xy')

    def get_device(self):
        return 'cuda' if torch.cuda.is_available else 'cpu'

    def train(self, dataset: Dataset, epochs: int):

        out_path = Path('data/models')
        best_model_path = out_path / str('best_steering_model_' + self.train_name + '.pth')
        best_loss = 1e9

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        train_losses = []
        test_losses = []
        test_losses_mae = []
        train_accs = []
        test_accs = []

        # live_plot = plt.gcf()
        # live_plot.show()
        # plt.title('Loss')
        # plt.xlabel("Epoch")
        # live_plot.canvas.draw()

        mae_loss = torch.nn.L1Loss()
        mae_loss.to(self.device)

        for epoch in tqdm(range(epochs)):

            # train the model
            self.model.train()
            train_loss = 0.0
            train_loss_mae = 0.0
            acc = 0
            for steps, (images, labels) in enumerate(iter(dataset.train)):

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                if self.is_classification:
                    loss = criterion(outputs, labels)
                    _, predicted_labels = outputs.max(1)
                    acc += (predicted_labels == labels).float().sum() / predicted_labels.shape[0]
                else:
                    loss = F.mse_loss(outputs, labels)
                    loss_mae = mae_loss(outputs, labels)

                train_loss += float(loss)
                train_loss_mae += float(loss_mae)

                loss.backward()
                optimizer.step()

            train_loss /= len(dataset.train)
            train_losses.append(train_loss)
            if self.is_classification:
                train_accuracy = acc / (steps + 1)
                train_accs.append(train_accuracy)

            # eval the model
            self.model.eval()
            test_loss = 0.0
            test_loss_mae = 0.0
            acc = 0
            for steps, (images, labels) in enumerate(iter(dataset.test)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    outputs = self.model(images)
                if self.is_classification:
                    loss = criterion(outputs, labels)
                    _, predicted_labels = outputs.max(1)
                    acc += (predicted_labels == labels).float().sum() / predicted_labels.shape[0]
                else:
                    loss = F.mse_loss(outputs, labels)
                    loss_mae = mae_loss(outputs, labels)

                test_loss += float(loss)
                test_loss_mae += float(loss_mae)

            test_loss /= len(dataset.test)
            test_losses.append(test_loss)
            test_losses_mae.append(test_loss_mae)
            if self.is_classification:
                test_accuracy = acc / (steps + 1)
                test_accs.append(test_accuracy)

                print((f'Epoch: {epoch}'
                       f'\tTrain loss: {train_loss:.4f}'
                       f'\tTrain Acc {train_accuracy:.4f}'
                       f'\tTest loss: {test_loss:.4f}'
                       f'\tTest Acc {test_accuracy:.4f}'))
            else:
                print((f'Epoch: {epoch}'
                       f'\tTrain loss (MSE): {train_loss:.4f}'
                       f'\tTest loss (MSE): {test_loss:.4f}'
                       f'\tTrain loss (MAE): {train_loss_mae:.4f}'
                       f'\tTest loss (MAE): {test_loss_mae:.4f}'))

            if test_loss < best_loss:
                #self.model = self.model.to('cpu')
                torch.save(self.model.state_dict(), str(best_model_path))
                best_loss = test_loss
                #self.model = self.model.to(self.device)

            # plt.plot(np.array(train_losses), 'r', label='train')
            # plt.plot(np.array(test_losses), 'b', label='test')
            # plt.pause(0.01)
            # live_plot.canvas.draw()

        loss_plot = out_path / str('training_result_' + self.train_name + '_loss.png')
        save_plot(loss_plot, train_losses, test_losses, test_losses_mae, metric='Loss', n=0)
        mean_e = sum(test_losses) / len(test_losses)
        mean_a = sum(test_losses_mae) / len(test_losses_mae)
        print(f'\n\n{mean_a}............{mean_e}\n\n')

        if self.is_classification:
            acc_plot = out_path / str('training_result_' + self.train_name + '_acc.png')
            save_plot(acc_plot, train_accs, test_accs, metric='Accuracy', n=1)


def get_dataset(dataset_path: str, labels_path: str, n_classes: int, variable: str = 'xy', test_percent: float = 0.1):
    dataset = Dataset(dataset_path, labels_path, n_classes, variable=variable)
    dataset.split_train_test(test_percent)    # percent of the dataset for testing
    dataset.to_dataloader(batch_size=32)

    return dataset

def change_head(model, out_features: int, type_: int = 0):
    if type_ == 0:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, out_features)
    elif type_ == 1:
        in_features = model.last_channel
        model.classifier[1] = torch.nn.Linear(in_features, out_features)

    return model


def get_model(model_name: str, pretrained: bool, n_classes: int):
    type_ = 0
    if 'resnet18' in model_name:
        model = torchvision.models.resnet18(pretrained=pretrained)
        type_ = 0
    elif 'resnet34' in model_name:
        model = torchvision.models.resnet34(pretrained=pretrained)
        type_ = 0
    elif 'mobilenet' in model_name:
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        type_ = 1

    return change_head(model, n_classes, type_)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default='resnet18', type=str,
                        help="Model architecture (resnet18, mobilenet_v2, etc.). Default: resnet18")
    parser.add_argument("--pretrained", default=True, type=bool,
                        help="Load the pretrained weight of the model or train from scratch. Default: True")
    parser.add_argument("--dataset-dir", required=True,
                        help="Location of image files.")
    parser.add_argument("--labels-file", required=True,
                        help="Location of labels file.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write out the artifacts.")
    parser.add_argument("--num-classes", required=False, type=int,
                        help="Model output size. Default: 2 for regression of x and y")
    parser.add_argument("--train-variable", default='xy', type=str,
                        help="Variable to be trained. Default: 'xy")
    parser.add_argument("--test-size", default=0.1, type=float,
                        help="Percentage of dataset used for test. Default: 0.1")
    parser.set_defaults(do_lower_case=True)
    args = parser.parse_args()

    model_name = args.model
    pretrained = args.pretrained
    dataset_path = args.dataset_dir
    labels_file = args.labels_file
    output_path = args.output_dir
    n_classes = args.num_classes
    variable = args.train_variable
    test_size = args.test_size

    dataset = get_dataset(dataset_path, labels_file, n_classes, variable, test_size)

    model = get_model(model_name, pretrained, n_classes)
    trainer = Trainer(model, variable + '_' + model_name, variable)

    trainer.train(dataset, epochs=70)


if __name__ == '__main__':
    main()
