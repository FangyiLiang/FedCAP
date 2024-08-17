import json

import numpy as np
from PIL import Image
import cv2
import os

class Mini_Imagenet:
    def __init__(self, root, train, transform=None):
        super(Mini_Imagenet, self).__init__()
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
        class_path = os.listdir(self.root)
        self.class_names=self.get_class_names(class_path)
        print(class_path)
        print(self.class_names)
        for i in range(len(class_path)):

            class_temp = os.path.join(self.root, class_path[i])

            img_path = os.listdir(class_temp)
            for j in range(len(img_path)):
                img_path_temp = os.path.join(class_temp, img_path[j])
                img = cv2.imread(img_path_temp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if j < 500:
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


