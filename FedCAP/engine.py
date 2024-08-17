
import copy
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
from tqdm import tqdm
import torch

import numpy as np
from clip import clip
from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
from torch.utils.data import DataLoader
from main import input_sentence_to_txt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def train_one_epoch(clip_model: torch.nn.Module,client_id:int,criterion,epoch, data_loader, optimizer: torch.optim.Optimizer,device: torch.device, max_norm: float = 0,set_training_mode=True,task_id=-1,args=None):
    clip_model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        clip_model.adaptation(target, device)
        ground_truth = torch.arange(len(input), dtype=torch.long, device=device)

        logits_per_image, logits_per_text =clip_model(input,clip_model.text_tokens,train=True)

        loss = (criterion(logits_per_image, ground_truth) + criterion(logits_per_text, ground_truth)) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])


    metric_logger.synchronize_between_processes()
    clip_model.eval()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,each_task_seen_classes, data_loader,device, task_id=-1, args=None):
    correct_predictions = 0
    total_images = 0
    device = args.device
    current_task_class_names = each_task_seen_classes[task_id]
    for input, target in tqdm(data_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in current_task_class_names]).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(input, text_inputs,train=False)

        similarity = logits_per_image.softmax(dim=-1)
        _, predicted_indices = similarity.topk(1)

        correct_predictions += (predicted_indices.squeeze() == target).sum().item()
        total_images += target.size(0)

    accuracy = correct_predictions / total_images
    return accuracy,correct_predictions,total_images


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module,ep_g, each_task_seen_classes,data_loader,
                      device, task_id=-1,args=None):
    each_acc=[]
    total_acc=0
    total_correct_predictions=0
    total_images_number=0
    truelabels_total=[]
    predictedlabels_total=[]
    for i in range(task_id + 1):

        acc,correct_predictions,total_images,truelabels,predictedlabels = evaluate_draw(model=model,each_task_seen_classes=each_task_seen_classes,data_loader=data_loader[i]['val'],
                              device=device, task_id=task_id, args=args)
        truelabels_total.append(truelabels)
        predictedlabels_total.append(predictedlabels)
        total_acc+=acc
        total_correct_predictions+=correct_predictions
        total_images_number+=total_images
        if i ==task_id:
            print(f"Task {task_id} Epoch {ep_g} with its prompt Accuracy: {acc * 100:.2f}%")
        temp_acc_format=f"{i}:{acc * 100:.2f}%"
        each_acc.append(temp_acc_format)
    avg_acc=total_correct_predictions/total_images_number
    print(f"Task {task_id} Epoch {ep_g} AverageAccuracy: {avg_acc * 100:.2f}%")
    avg_acc_format=f"{avg_acc * 100:.2f}%"
    acc_list=[]
    acc_list.append(avg_acc_format)
    acc_list.extend(each_acc)
    truelabels_total = np.concatenate(truelabels_total)
    predictedlabels_total = np.concatenate(predictedlabels_total)
    plot_confusion_matrix(truelabels_total, predictedlabels_total, 100,save_path='confusion_matrix_task{}'.format(task_id))
    return acc_list

def evaluate_draw(model: torch.nn.Module,each_task_seen_classes, data_loader,device, task_id=-1, args=None):
    correct_predictions = 0
    total_images = 0
    device = args.device
    true_labels = []
    predicted_labels = []
    current_task_class_names = each_task_seen_classes[task_id]
    for input, target in tqdm(data_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in current_task_class_names]).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(input, text_inputs)

        similarity = logits_per_image.softmax(dim=-1)
        _, predicted_indices = similarity.topk(1)

        true_labels.extend(target.cpu().numpy())
        predicted_labels.extend(predicted_indices.squeeze().cpu().numpy())

        correct_predictions += (predicted_indices.squeeze() == target).sum().item()
        total_images += target.size(0)

    accuracy = correct_predictions / total_images
    return accuracy,correct_predictions,total_images,np.array(true_labels),np.array(predicted_labels)




def train_and_evaluate(client,clip_model,client_id,task_id, criterion,old_client):
    if client_id in old_client:
        client.before_train(task_id, 0)
    else:
        client.before_train(task_id, 1)
    sentence="base_scope:"+str(sorted(client.base_scope))
    print("base_scope:",sorted(client.base_scope))

    data_loader = client.data_loader
    device = client.device
    args = client.args
    task_id_current = client.current_class

    if task_id == 0:
        use_mean_update_key(client_id,clip_model, data_loader[task_id]['train'], args)
    optimizer = create_optimizer(args, clip_model)

    for epoch in range(args.epochs_local):
        train_stats = train_one_epoch(clip_model=clip_model,
                                      client_id=client_id,
                                      criterion=criterion,
                                      epoch=epoch,
                                      data_loader=data_loader[task_id_current]['train'],
                                      optimizer=optimizer,
                                      device=args.device,
                                      max_norm=args.clip_grad,
                                      set_training_mode=True,
                                      task_id=task_id_current,
                                      args=args)


def use_mean_update_key(client_id,model, data_loader, args):

    mean_shape=(1,768)
    class_sum = {class_label: torch.zeros(mean_shape) for class_label in range(args.size)}
    class_count = {class_label: 0 for class_label in range(args.size)}

    for input, target in data_loader:
        with torch.no_grad():
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            input = model.clip_model.visual.conv1(input)
            input = input.reshape(input.shape[0], input.shape[1], -1)
            input = input.permute(0, 2, 1)
            input_mean=torch.mean(input, dim=1)
            for i in range(len(target)):
                class_sum[target[i].item()] += input_mean[i][:].detach().cpu()
                class_count[target[i].item()] += 1

    class_means = {class_label: class_sum[class_label] / class_count[class_label]  for class_label in range(60) if class_count[class_label] != 0}

    for class_label, mean_vector in class_means.items():
        sentence = f"Class {class_label} Mean: {mean_vector[0][0]}"

        mean_vector=mean_vector.to(args.device)
        if model.prompt.key_list[class_label]==0:
            model.prompt.key.data[class_label]=torch.nn.Parameter(mean_vector)
            model.prompt.key_list[class_label]=1





def client_evaluate(clip_model,client_id,each_task_seen_classes,epoch,data_loader,task_id,args):
    correct_predictions = 0
    total_images = 0
    device = args.device
    current_task_class_names=each_task_seen_classes[task_id]
    for input, target in tqdm(data_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in current_task_class_names]).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(input, text_inputs)

        similarity = logits_per_image.softmax(dim=-1)
        _, predicted_indices = similarity.topk(1)

        correct_predictions += (predicted_indices.squeeze() == target).sum().item()
        total_images += target.size(0)

    accuracy = correct_predictions / total_images



def evaluate_all(clip_model,args, data_loader):
    correct_predictions = 0
    total_images = 0
    device=args.device
    for input, target in tqdm(data_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in clip_model.class_names]).to(device)


        with torch.no_grad():
            logits_per_image, logits_per_text =clip_model(input,text_inputs)

        similarity=logits_per_image.softmax(dim=-1)
        _, predicted_indices = similarity.topk(1)


        correct_predictions += (predicted_indices.squeeze() == target).sum().item()
        total_images += target.size(0)

    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy*100


