import glob
import os

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

import pandas as pd

def _lookup_label(df: pd.DataFrame, id_img: str, variable: str):
    """ Helper function to search for labels in a dataframe given an id and the labels column names

    Parameters
    ----------
        df: Dataframe with all the labels information
        id_img: id of the image from which you want to obtain the labels
        variable: name of the column where the wanted labels are.
    """
    res = []
    for char in variable:
        res.append(df.loc[df['id'] == id_img][char].item())

    return res

def get_labels(variable: str, df: pd.DataFrame, n_classes: int, image_path: str):
    """ Get label of the image given from the labels file given the column names
    Parameters
    ----------
        variable: name of the columns where the wanted label are
        df: Dataframe with all the labels information
        image_path: path of the image from which you want to obtain the labels.

    Retruns
    -------
        list of labels
    """
    id_img = image_path.split('\\')[-1].split('_')[-1].split('.')[0]
    labels = _lookup_label(df, id_img, variable)
    if variable == 'xy':
        computed_labels = [(float(int(value)) - 50.0) / 50.0 for value in labels]
        return computed_labels
    else:
        # one_hot_labels = np.zeros(n_classes)
        # one_hot_labels[labels] = 1
        return labels[0]

class _Dataset(torch.utils.data.Dataset):

    """ This class acts like a iterable data structure to retrieve samples of the dataset on demand. """

    def __init__(self, directory: str, labels_file: str, n_classes: int, variable: str = 'xy'):
        self.directory = directory
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        self.labels = pd.read_csv(labels_file, sep='\t')
        self.variable = variable
        self.n_classes = n_classes

    def _transform_image(self, image):
        """ Make color and geometry transformations to the given image. """

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        labels = get_labels(self.variable, self.labels, self.n_classes, image_path)

        image = self._transform_image(image)

        if self.variable == 'xy':
            return image, torch.tensor(labels).float()
        else:
            return image, torch.tensor(labels, dtype=torch.long)


class Dataset(object):

    def __init__(self, directory: str, labels_path: str, n_classes: int, variable: str = None):

        self.data = _Dataset(directory, labels_path, n_classes, variable=variable)
        self.train = None
        self.test = None

    def split_train_test(self, test_percent):
        test_size = int(test_percent * len(self.data))
        self.train, self.test = torch.utils.data.random_split(self.data, [len(self.data) - test_size, test_size])

        return self.train, self.test

    def to_dataloader(self, batch_size: int):
        self.train = self.__convert_to_dataloader(self.train, batch_size)
        self.test = self.__convert_to_dataloader(self.test, batch_size)

    def __convert_to_dataloader(self, data, b_size):
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=b_size,
            shuffle=True,
            num_workers=0
        )

        return data_loader
