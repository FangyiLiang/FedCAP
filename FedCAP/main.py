# from model import baseClip
import sys
from typing import Iterable
import copy
import torchvision.datasets
from collections import OrderedDict

import vitmodel
# from option import args_parser
# from baseClip import baseClip
# from models import BaseClip, VisionTransformer1,VisionTransformer2
from clipmodel import ClipModel
from torch.utils.data import DataLoader
import os.path as osp
import os
from PIL import Image
from pathlib import Path
import numpy as np
import torch
import random
import clip
from clip.clip import _transform
import torch.nn as nn
import math
import torchvision
import matplotlib.pyplot as plt
import argparse
import torch.backends.cudnn as cudnn
from dataloader import FewShotContinualDataloader
from timm.optim import create_optimizer
from trainers.FedFSCIL import load_clip_to_cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
import datetime
import time
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from trainers.FedFSCIL import FedFSCIL
from engine import *
from GLFC import GLFC_model
from Fed_utils import *
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_args_parser():
    parser=argparse.ArgumentParser('FedFSCIL training and evaluation configs', add_help=False)

    parser.add_argument('--batch-size', default=16, type=int, help='Batch size per device')
    # parser.add_argument('--epochs', default=1, type=int)
    # Continual learning parameters
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=30, help='initial number of clients')
    parser.add_argument('--local_clients', type=int, default=10, help='number of selected clients each round')
    parser.add_argument('--epoch',default=30,type=int,help='total count of data used time')
    parser.add_argument('--epochs_local',default=5,type=int,help='total count of data used time')
    parser.add_argument('--epochs_global',default=90,type=int,help='total count of data used time')

    # Prompt parameters
    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--size', default=60, type=int,)
    parser.add_argument('--length', default=5,type=int, )
    parser.add_argument('--top_k', default=5, type=int, )
    parser.add_argument('--initializer', default='uniform', type=str,)
    parser.add_argument('--prompt_key', default=True, type=bool,)
    parser.add_argument('--prompt_init', default='uniform', type=str)

    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=False, type=bool)
    parser.add_argument('--shared_prompt_pool', default=False, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)

    # Model parameters
    parser.add_argument('--trainer', default='FedFSCIL', type=str, help="name of trainer")
    parser.add_argument('--trainer_prec', default='fp32', choices=['fp16', 'fp32', 'amp'],type=str, help="trainer 精度")
    parser.add_argument('--model_name', default='ViT-B/16', type=str, help="model name of ViT")
    parser.add_argument('--prompt_template', default='a photo of a {}', type=str, help="template of prompt")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',help='Drop path rate (default: 0.)')
    parser.add_argument('--jit', default=False, type=bool, help="clip model need jit or not")

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER',help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True,help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data-path', default='../datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='cifar100',choices=['cifar100', 'mini-imagenet', 'tiny-imagenet-200'], type=str, help='dataset name')
    parser.add_argument('--num_tasks', default=9, type=int, help='number of sequential tasks')
    parser.add_argument('--classes_per_task', default=10, type=int, help='number of classes per task')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Misc parameters
    parser.add_argument('--print_freq', type=int, default=70, help='The frequency of printing')

    return parser

def input_sentence_to_csv(file_path, mode,sentence):
    with open(file_path, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([sentence])

def append_list_to_csv(file_path, input_list):
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(input_list)

def input_sentence_to_txt(file_path, mode,sentence):
    f = open(file_path, mode)
    f.write(sentence)
    f.write('\n')
    f.close()

def count_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    mem_params_mb = trainable_params * 4 / (1024 * 1024)

    return trainable_params, mem_params_mb

def main(args):
    file_path = './output/result.csv'
    start_sentence='use prompt pool size=60,top_k=5,save model'
    input_sentence_to_csv(file_path,'w',start_sentence)

    utils.init_distributed_mode(args)

    device=torch.device(args.device)

    seed=args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    continual_dataloader=FewShotContinualDataloader(args)
    data_loader,each_task_seen_classes_global,_=continual_dataloader.create_dataloader("global")
    for i in range(len(each_task_seen_classes_global)):
        print(i," ",len(each_task_seen_classes_global[i]))
    class_names=continual_dataloader.class_names
    print(continual_dataloader.class_names_pair)

    print(f"Creating global model: {args.model_name}")
    temp_sentence=f"Creating global model: {args.model_name}"
    input_sentence_to_csv(file_path,'a',temp_sentence)
    model_g=FedFSCIL(args,class_names)

    original_model_g=copy.deepcopy(model_g)
    original_model_g.to(device)
    model_g.to(device)

    model_g.set_parameters_grad()

    enabled = set()
    for name, param in model_g.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    temp_sentence=f"Parameters to be updated: {enabled}"

    trainable_count, mem_size = count_trainable_params(model_g)

    print(f"Number of trainable parameters: {trainable_count:,}")
    print(f"Memory used by trainable parameters: {mem_size:.2f} MB")

    total_params = sum(p.numel() for p in model_g.parameters())
    print(f"Total number of parameters: {total_params:,}")

    input_sentence_to_csv(file_path,'a',temp_sentence)

    num_clients = args.num_clients
    old_client_0 = []
    old_client_1 = [i for i in range(args.num_clients)]
    new_client = []
    models = []

    for i in range(130):
        model_temp = GLFC_model(args, i, continual_dataloader)
        models.append(model_temp)
    n_parameters = sum(p.numel() for p in model_g.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    temp_sentence=f"number of params:{n_parameters}"
    input_sentence_to_csv(file_path,'a',temp_sentence)


    criterion = torch.nn.CrossEntropyLoss().to(device)
    print(f"Start federated training")
    start_time = time.time()

    old_task_id = -1
    models_local = [copy.deepcopy(model_g) for i in range(args.local_clients)]
    output_list = []
    for ep_g in range(args.epochs_global):
        output_list=[]
        task_id = ep_g // (args.epochs_global // args.num_tasks)
        temp_sentence=f"Task {task_id}"
        output_list.append(temp_sentence)
        temp_epoch=ep_g % (args.epochs_global // args.num_tasks)
        temp_sentence=f"epoch {temp_epoch}"
        output_list.append(temp_sentence)

        if task_id != old_task_id and old_task_id != -1:
            overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
            new_client = [i for i in range(overall_client, overall_client + 100 // (args.nb_classes // args.classes_per_task))]
            old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
            old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
            num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
            print("old_client_0:",old_client_0)

        n_parameters = sum(p.numel() for p in model_g.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        w_local_prompt = []
        w_local_key = []
        w_local_key_list=[]
        w_metaNet_hw=[]
        w_metaNet_hb=[]
        w_metaNet_ow=[]
        w_metaNet_ob=[]
        clients_index = random.sample(range(num_clients), args.local_clients)
        print('select part of clients to conduct local training')
        print(clients_index)
        output_list.extend(clients_index)

        for j, c in enumerate(clients_index):
            models_local[j] = copy.deepcopy(model_g)
            sentence='federated global round: {}, client: {}, task_id: {}'.format(ep_g, c, task_id)
            print(sentence)
            input_sentence_to_txt('./output/client/client{}mean_check.txt'.format(c),'a',sentence)
            train_and_evaluate(models[c], models_local[j], c, task_id, criterion, old_client_0)
            model_local = models_local[j].state_dict()
            model_local_key=model_local['prompt.key']
            model_local_key_list=models_local[j].prompt.key_list
            model_local_prompt=model_local['prompt.prompt']
            model_local_hw=model_local['metaNet.hidden_layer.weight']
            model_local_hb=model_local['metaNet.hidden_layer.bias']
            model_local_ow=model_local['metaNet.output_layer.weight']
            model_local_ob=model_local['metaNet.output_layer.bias']
            w_local_key.append(model_local_key)
            w_local_prompt.append(model_local_prompt)
            w_local_key_list.append(model_local_key_list)
            w_metaNet_hw.append(model_local_hw)
            w_metaNet_hb.append(model_local_hb)
            w_metaNet_ow.append(model_local_ow)
            w_metaNet_ob.append(model_local_ob)

        # every participant update their classes
        print('every participant update their classes...')
        participant_update(models, num_clients, old_client_0, task_id, clients_index)
        print('updating finishes')

        print('federated aggregation...')
        state_dict = model_g.state_dict()

        w_key_new,w_key_list_new=FedKey(state_dict['prompt.key'],model_g.prompt.key_list,w_local_key,w_local_key_list)
        w_prompt_new = FedPrompt2(w_local_prompt)
        w_metaNet_hw_new = FedPrompt2(w_metaNet_hw)
        w_metaNet_hb_new = FedPrompt2(w_metaNet_hb)
        w_metaNet_ow_new = FedPrompt2(w_metaNet_ow)
        w_metaNet_ob_new = FedPrompt2(w_metaNet_ob)

        state_dict['prompt.key'] = w_key_new
        model_g.prompt.key_list=w_key_list_new
        state_dict['prompt.prompt'] = w_prompt_new
        state_dict['metaNet.hidden_layer.weight']=w_metaNet_hw_new
        state_dict['metaNet.hidden_layer.bias']=w_metaNet_hb_new
        state_dict['metaNet.output_layer.weight']=w_metaNet_ow_new
        state_dict['metaNet.output_layer.bias']=w_metaNet_ob_new
        model_g.load_state_dict(state_dict)
        acc_list=evaluate_till_now(model=model_g,ep_g=(ep_g % (args.epochs_global // args.num_tasks)), each_task_seen_classes=each_task_seen_classes_global,data_loader=data_loader, device=device,task_id=task_id, args=args)
        output_list.extend(acc_list)

        if args.output_dir:
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id))
            state_dict = {
                'model': model_g.state_dict(),
                'epoch': args.epochs_local,
                'args': args,
            }
            torch.save(state_dict, checkpoint_path)


        old_task_id = task_id
        append_list_to_csv(file_path,output_list)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")
    temp_sentence=f"Total training time: {total_time_str}"
    input_sentence_to_csv(file_path,'a',temp_sentence)




if __name__ == '__main__':

    parser = argparse.ArgumentParser('FedFSCIL training and evaluation configs',parents=[get_args_parser()])
    args=parser.parse_args()
    print(args)
    args.output_dir=os.path.join(args.output_dir,args.dataset,'seed{}'.format(args.seed))
    print("outputdir:",args.output_dir)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)

    if args.dataset == 'cifar100':
        print(args.dataset)
    elif args.dataset == 'mini-imagenet':
        print(args.dataset)
    elif args.dataset == 'tiny-imagenet-200':
        print(args.dataset)
        args.epochs_global=110
        args.num_tasks=11
        args.classes_per_task=20
        args.size=100
    else:
        raise NotImplementedError(f"Not supported dataset:{args.dataset}")
    main(args)
    sys.exit(0)








