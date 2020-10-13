import json
import csv
import os
import glob


"""
# xy
[
    {
        id: 'ccccc',
        x: 50,
        y: 85,
    },
    ...
]


# vw
[
    {
        id: 'ccccc',
        v: 1,
        w: 4,
    },
    ...
]
"""



def create_classification_file():
    def get_v(path):
        return int(path.split('_')[1])

    def get_w(path):
        return int(path.split('_')[2])

    dataset_path = 'data/datasets/class5_7/'

    samples = []
    for filename in glob.glob(dataset_path + '*.jpg'):

        dct = {'id': '', 'v': 0, 'w': 0}
        image_name = os.path.basename(filename)
        id_img = image_name.split('_')[-1].split('.')[0]
        v = get_v(image_name)
        w = get_w(image_name)
        dct['id'] = id_img
        dct['v'] = v
        dct['w'] = w
        samples.append(dct)

    json_name = json.dumps(samples)
    with open('data/datasets/classification_5v7w.json', 'w') as f:
        f.write(json_name)
    
    keys = samples[0].keys()
    with open('data/datasets/classification_5v7w.tsv', 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
        dict_writer.writeheader()
        dict_writer.writerows(samples)  



def create_regression_file():

    def get_x(path):
        x = int(path.split('_')[1])
        return f'{x:03}'

    def get_y(path):
        y = int(path.split('_')[2])
        return f'{y:03}'

    dataset_path = 'data/datasets/original/'

    samples = []
    for filename in glob.glob(dataset_path + '**/*.jpg'):

        dct = {'id': '', 'x': 0, 'y': 0}
        image_name = os.path.basename(filename)
        id_img = image_name.split('_')[-1].split('.')[0]
        x = get_x(image_name)
        y = get_y(image_name)
        dct['id'] = id_img
        dct['x'] = x
        dct['y'] = y
        samples.append(dct)

    json_name = json.dumps(samples)
    with open('data/datasets/regression.json', 'w') as f:
        f.write(json_name)

    keys = samples[0].keys()
    with open('data/datasets/regression.tsv', 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys, delimiter='\t')
        dict_writer.writeheader()
        dict_writer.writerows(samples)  


create_regression_file()
create_classification_file()