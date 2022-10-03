import torch
import torch.nn as nn
from torch import optim
from util import evaluate_prediction
import random
import torch.nn.functional as F
import os
import numpy as np
from model.Loss import WeightBCEWithLogitsLoss
import copy
from torch.utils.data import DataLoader, RandomSampler

class Reptile(nn.Module):
    def __init__(self, args, model):
        super(Reptile, self).__init__()
        self.args = args
        self.model = model
        if self.args.cuda:
            self.model.to(self.args.device)
        self.model_state=self.model.state_dict()
        self.query_node_num = args.query_node_num
        self.meta_lr = args.meta_lr
        self.update_lr = args.learning_rate
        self.num_shots = args.num_shots
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.epsilon= 1/self.update_step
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.meta_lr, weight_decay=self.args.weight_decay)
        self.criterion = WeightBCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,0.9, last_epoch=-1)


    def train_Reptile(self, train_tasks):
        support_batches = [ task.get_batch() for task in train_tasks]
        for epoch in range(self.args.epochs):
            random.shuffle(support_batches)
            new_vars=[]
            random_support_batch=support_batches
            num_tasks = len(random_support_batch)
            taskloss=0
            for i in range(num_tasks):
                model=copy.deepcopy(self.model)
                model.train()
                optimizer = optim.Adam(model.parameters(), lr = self.update_lr, weight_decay=self.args.weight_decay)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
                support_batch = random_support_batch[i]
                if self.args.cuda:
                        support_batch = support_batch.to(self.args.device)
                for k in range(self.update_step):
                    output, mask = model(support_batch)
                    output = output.squeeze()
                    loss = self.criterion(output, support_batch.y, mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                taskloss=taskloss+loss.item()
                new_vars.append(model.state_dict())
            taskloss=taskloss/num_tasks
            new_vars=average_vars(new_vars)
            self.model_state = interpolate_vars(self.model_state, new_vars, self.epsilon)
            self.model.load_state_dict(self.model_state)


    def evaluate_Reptile(self, test_tasks):
        support_batches = [task.get_support_batch(isRep=True) for task in test_tasks]
        query_batches = [task.get_query_batch(isRep=True) for task in test_tasks]
        num_tasks = len(query_batches)
        predict_all=[]
        target_all=[]
        self.model=self.model.cpu()
        for i in range(num_tasks):
            support_batch = support_batches[i]
            query_batch = query_batches[i]
            if self.args.cuda:
                support_batch = support_batch
                query_batch = query_batch
            for k in range(self.update_step_test):
                self.model.train()
                output, mask = self.model(support_batch)
                output = output.squeeze()
                loss = self.criterion(output, support_batch.y, mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            query_output,_ = self.model(query_batch)
            query_output = query_output.squeeze()
            pred = torch.sigmoid(query_output)
            pred = torch.where(pred > 0.5, 1, 0)

            pred, targets = pred.cpu().detach().numpy(), query_batch.y.cpu().detach().numpy()
            acc, precision, recall, f1 = evaluate_prediction(pred, targets)
            pred=pred.reshape(-1,1)
            targets=targets.reshape(-1,1)
            print("Test Task-{}: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(i, acc, precision, recall, f1))
            if len(predict_all)==0:
                predict_all=pred
                target_all=targets
            else:
                predict_all = np.vstack((predict_all,pred))
                target_all = np.vstack((target_all,targets))
        acc, precision, recall, f1 = evaluate_prediction(predict_all, target_all)
        print("Reptile Test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(acc, precision, recall, f1))

def average_vars(dict_list):
    """
    Averrage of list of state_dicts.
    """
    for param_tensor in dict_list[0]:
        for i in range(1, len(dict_list)):
            dict_list[0][param_tensor] = dict_list[0][param_tensor] + dict_list[i][param_tensor]
        dict_list[0][param_tensor] = dict_list[0][param_tensor]/len(dict_list)

    average_var = dict_list[0]

    return average_var

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    for param_tensor in new_vars:
        new_vars[param_tensor]  = old_vars[param_tensor] + (new_vars[param_tensor] - old_vars[param_tensor]) * epsilon
    return new_vars

