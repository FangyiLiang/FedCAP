# --------------------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# --------------------------------------------------------
import os
import random
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Lambda
from clip.clip import _transform
from PIL import Image
import copy
from mini_imagenet import *
from tiny_imagenet import *
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



class FewShotContinualDataloader:
    def __init__(self, args):
        self.args = args
        if not os.path.exists(self.args.data_path):
            os.makedirs(self.args.data_path)
        self._get_dataset(self.args.dataset)
        self.name=self.args.dataset

    def _get_dataset(self, name):
        input_resolution = 224
        if name == 'cifar100':
            root = self.args.data_path
            self.dataset_train = datasets.CIFAR100(root=root, train=True, download=True,
                                                   transform=_transform(input_resolution))
            self.dataset_val = datasets.CIFAR100(root=root, train=False, transform=_transform(input_resolution))
            self.args.nb_classes = 100
            cifar100_class_names = datasets.CIFAR100(root=root, train=True, download=True).classes
            self.class_names_pair = []
            self.class_names = []
            for label, class_names in enumerate(cifar100_class_names):
                self.class_names_pair.append({label: class_names})
                self.class_names.append(class_names)
        elif name == 'mini-imagenet':
            root = self.args.data_path + self.args.dataset
            self.dataset_train = Mini_Imagenet(root=root, train=True, transform=_transform(input_resolution))
            self.dataset_val = Mini_Imagenet(root=root, train=False, transform=_transform(input_resolution))
            self.args.nb_classes = 100
            self.class_names_pair = []
            self.class_names=self.dataset_train.class_names
            print("mini-imagenet",self.class_names,type(self.class_names))
        elif name == 'tiny-imagenet-200':
            root = self.args.data_path + self.args.dataset
            self.dataset_train = Tiny_Imagenet(root=root, train=True, transform=_transform(input_resolution))
            self.dataset_val = Tiny_Imagenet(root=root, train=False, transform=_transform(input_resolution))
            self.args.nb_classes = 200
            self.class_names_pair = []
            self.class_names=self.dataset_train.class_names
            print("tiny-imagenet-200",self.class_names,type(self.class_names))

        else:
            raise NotImplementedError(f"Not supported dataset: {self.args.dataset}")

        self.labels = [i for i in range(self.args.nb_classes)]
        if self.args.shuffle:
            random.shuffle(self.labels)

    def split(self, usage):
        dataloader = []
        labels = self.labels
        each_task_seen_classes = []
        current_name = []
        base_class_num=60
        train_split_indices = []
        test_split_indices = []
        few_shot_num_tasks=8
        few_shot_classes_per_task=5
        each_class_few_shot_num=5
        if self.name == 'cifar100':
            base_class_num=60
            few_shot_num_tasks=8
            few_shot_classes_per_task = 5
            each_class_few_shot_num = 5
        elif self.name == 'mini-imagenet':
            base_class_num=60
            few_shot_num_tasks=8
            few_shot_classes_per_task = 5
            each_class_few_shot_num = 5
        elif self.name == 'tiny-imagenet-200':
            base_class_num=100
            few_shot_num_tasks=10
            few_shot_classes_per_task = 10
            each_class_few_shot_num = 5

        scope = labels[:base_class_num]
        for i in range(len(scope)):
            current_name.append(self.class_names[scope[i]])
        temp = copy.deepcopy(current_name)
        each_task_seen_classes.append(temp)

        if usage == 'local':
            scope = random.sample(scope, int(base_class_num * 0.6))
            base_scope=scope
        elif usage == 'global':
            scope = random.sample(scope, base_class_num)
            base_scope=scope
        else:
            raise ValueError(f'Invalid usage={usage}')
        labels = labels[base_class_num:]


        for k in range(len(self.dataset_train.targets)):
            if int(self.dataset_train.targets[k]) in scope:
                train_split_indices.append(k)


        for h in range(len(self.dataset_val.targets)):
            if int(self.dataset_val.targets[h]) in scope:
                test_split_indices.append(h)


        dataset_train, dataset_val = Subset(self.dataset_train, train_split_indices), Subset(self.dataset_val,
                                                                                             test_split_indices)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})


        for _ in range(few_shot_num_tasks):
            train_split_indices = []
            test_split_indices = []

            scope = labels[:few_shot_classes_per_task]
            for i in range(len(scope)):
                current_name.append(self.class_names[scope[i]])
            temp = copy.deepcopy(current_name)
            each_task_seen_classes.append(temp)

            if usage == 'local':
                scope = random.sample(scope, int(few_shot_classes_per_task * 0.6))
            elif usage == 'global':

                scope = random.sample(scope, few_shot_classes_per_task)
            else:
                raise ValueError(f'Invalid usage={usage}')

            labels = labels[few_shot_classes_per_task:]


            temp_train_split_indices=[]
            for i in range(len(scope)):
                one_class_train_split_indices=[]
                for k in range(len(self.dataset_train.targets)):
                    if int(self.dataset_train.targets[k]) == int(scope[i]):
                        one_class_train_split_indices.append(k)
                temp_train_split_indices.append(one_class_train_split_indices)

            for class_indices in temp_train_split_indices:
                random.shuffle(class_indices)
                train_split_indices.extend(class_indices[:each_class_few_shot_num])

            for h in range(len(self.dataset_val.targets)):
                if int(self.dataset_val.targets[h]) in scope:
                    test_split_indices.append(h)



            dataset_train, dataset_val = Subset(self.dataset_train, train_split_indices), Subset(self.dataset_val,
                                                                                                 test_split_indices)

            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)


            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
            )


            dataloader.append({'train': data_loader_train, 'val': data_loader_val})

        return dataloader, each_task_seen_classes,base_scope

    def create_dataloader(self, usage):
        dataloader, each_task_seen_classes,base_scope = self.split(usage)

        return dataloader, each_task_seen_classes,base_scope
