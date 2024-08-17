import torch
import torch.nn as nn
import copy

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg


def FedPrompt(models, count_prompts):
    w_prompt = copy.deepcopy(models[0]['prompt.prompt']).zero_()
    w_prompt_key = copy.deepcopy(models[0]['prompt.prompt_key']).zero_()
    for i in range(0, len(models)):
        prompt = models[i]['prompt.prompt']
        prompt_key = models[i]['prompt.prompt_key']
        count_prompts[i] = torch.tensor(count_prompts[i])
        _, key_idxs = torch.sort(count_prompts[i])
        for j, key_id in enumerate(key_idxs):
            w_prompt[j] += prompt[key_id]
            w_prompt_key[j] += prompt_key[key_id]
    w_prompt = torch.div(w_prompt, len(models))
    w_prompt_key = torch.div(w_prompt_key, len(models))

    w_avg = copy.deepcopy(models[0])
    w_avg['prompt.prompt'] = w_prompt
    w_avg['prompt.prompt_key'] = w_prompt_key
    for k in w_avg.keys():
        if k != 'prompt.prompt' and k != 'prompt.prompt_key':
            for i in range(1, len(models)):
                w_avg[k] += models[i][k]
            w_avg[k] = torch.div(w_avg[k], len(models))

    return w_avg


def FedPrompt2(models):
    w_avg = copy.deepcopy(models[0])
    for i in range(1, len(models)):
        prompt = models[i]
        w_avg += prompt
    w_avg = torch.div(w_avg, len(models))
    return w_avg

def FedKey(model_g_key,model_g_key_list,local_key,local_key_list):
    for i in range(len(local_key)):
        temp_key=local_key[i]
        temp_key_list=local_key_list[i]
        for j in range(len(temp_key_list)):
            if model_g_key_list[j]==0 and temp_key_list[j]==1:
                model_g_key[j]=temp_key[j]
                model_g_key_list[j]=1
    return model_g_key,model_g_key_list

def participant_update(clients, num, old_client, task_id, clients_index):
    for index in range(num):
        #如果30个客户端中，index不是当前参数训练的客户端
        if index not in clients_index:
            #第一个task没有
            if index in old_client:

                clients[index].before_train(task_id, 0)
            else:
                clients[index].before_train(task_id, 1)

