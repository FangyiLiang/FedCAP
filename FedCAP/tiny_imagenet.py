import json

import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd

class Tiny_Imagenet:
    def __init__(self, root, train, transform=None):
        super(Tiny_Imagenet, self).__init__()
        self.transform = transform
        self.data = None
        self.targets = None
        self.root = root
        self.train = train
        self.class_names=[]
        self.get_data()

    def get_class_names(self,class_path):
        names_raw=[]
        class_names=[]
        with open('./datasets/imagenet_class_index.json', 'r') as file:
            json_data = file.read()
        data = json.loads(json_data)
        for key,value in data.items():
            species_id, species_name = value
            names_raw.append({species_id:species_name})
        for species_id in class_path:
            for name_dict in names_raw:
                if species_id in name_dict:
                    class_names.append(name_dict[species_id])
                    break
        return class_names
    def get_data(self):
        train_list_img, train_list_label, test_list_img, test_list_label = [], [], [], []
        train_path = os.path.join(self.root, 'train/')
        class_path = os.listdir(train_path)
        self.class_names=self.get_class_names(class_path)
        print(class_path)
        print(self.class_names)
        for i in range(len(class_path)):
            class_temp = os.path.join(train_path, class_path[i], 'images/')
            img_path = os.listdir(class_temp)
            for j in range(len(img_path)):
                img_path_temp = os.path.join(class_temp, img_path[j])
                img = cv2.imread(img_path_temp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                train_list_img.append(img)
                train_list_label.append(i)

        classes2id = {
            class_name: class_id
            for class_id, class_name in enumerate(os.listdir(os.path.join(self.root, "train")))
        }

        with open(os.path.join(self.root, "val", "val_annotations.txt")) as f:
            for line in f:
                split_line = line.split("\t")

                path, class_label = split_line[0], split_line[1]
                class_id = classes2id[class_label]
                img_path_temp = os.path.join(self.root, "val", "images", path)
                img = cv2.imread(img_path_temp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                test_list_img.append(img)
                test_list_label.append(class_id)

        train_list_img, test_list_img = np.asarray(train_list_img), np.asarray(test_list_img)

        if self.train:
            self.data, self.targets = train_list_img, train_list_label
        else:
            self.data, self.targets = test_list_img, test_list_label



    def get_data_delete(self):
        train_list_img, train_list_label, test_list_img, test_list_label = [], [], [], []
        train_path = os.path.join(self.root, 'train/')
        class_path = os.listdir(train_path)

        test_path = os.path.join(self.root, 'val/')
        val_data = pd.read_csv(test_path + 'val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])


        for i in range(len(class_path)):
            class_temp = os.path.join(train_path, class_path[i], 'images/')
            img_path = os.listdir(class_temp)
            for j in range(len(img_path)):
                img_path_temp = os.path.join(class_temp, img_path[j])
                img = cv2.imread(img_path_temp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if j < 450:
                    train_list_img.append(img)
                    train_list_label.append(i)
                else:
                    test_list_img.append(img)
                    test_list_label.append(i)

        train_list_img, test_list_img = np.asarray(train_list_img), np.asarray(test_list_img)

        if self.train:
            self.data, self.targets = train_list_img, train_list_label
        else:
            self.data, self.targets = test_list_img, test_list_label

    def getItem(self, index):
        img, target = Image.fromarray(self.data[index]), self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __getitem__(self, index):
        return self.getItem(index)

    def __len__(self):
        return len(self.data)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]


