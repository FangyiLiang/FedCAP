import copy
import torch
import numpy as np
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

class GLFC_model:

    def __init__(self, args, client_id, continual_dataloader):

        super(GLFC_model, self).__init__()
        self.global_batch_size = None
        self.args = copy.deepcopy(args)
        self.client_id = client_id
        self.device = torch.device(args.device)
        self.data_loader,self.each_task_seen_classes,self.base_scope = continual_dataloader.create_dataloader('local')
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1

    def learning_rate(self):
        if self.args.unscale_lr:
            self.global_batch_size = self.args.batch_size * self.args.world_size
        else:
            self.global_batch_size = self.args.batch_size
        self.args.lr = self.args.lr * self.global_batch_size / 256.0

    def save_model(self, model):

        model_local = copy.deepcopy(model)
        torch.save(model_local, self.args.output_dir + '/clients/client{}.pth'.format(self.client_id))

    def load_model(self):

        return torch.load(self.args.output_dir + '/clients/client{}.pth'.format(self.client_id))

    def before_train(self, task_id_new, group):
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
                self.current_class = task_id_new
            else:
                self.last_class = None

