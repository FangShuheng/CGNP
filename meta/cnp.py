from copy import deepcopy
from operator import index
from pyexpat import model
import threading
import torch
import torch.nn as nn
from torch import optim
from model.Loss import WeightBCEWithLogitsLoss
from util import evaluate_prediction
import random
import torch.nn.functional as F
import os
import numpy as np
import copy

class CNP(nn.Module):
    def __init__(self, args, model):
        super(CNP, self).__init__()
        self.args = args
        self.model = model
        self.maxmodel=model
        self.learning_rate = args.learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience)
        self.criterion = WeightBCEWithLogitsLoss()
        if self.args.cuda:
            self.model.to(self.args.device)

    def train_cnp(self, train_tasks, valid_tasks, test_tasks):
        support_batches = [ task.get_support_batch() for task in train_tasks]
        query_batches = [ task.get_query_batch() for task in train_tasks]
        for epoch in range(self.args.epochs):
            self.model.train()
            # manually shuffle tasks
            tasks_batches = list(zip(support_batches, query_batches))
            random.shuffle(tasks_batches)
            support_batches, query_batches = zip(*tasks_batches)
            epoch_loss = 0.0
            for i in range(len(support_batches)):
                self.optimizer.zero_grad()
                support_batch = support_batches[i]
                query_batch = query_batches[i]
                if self.args.cuda:
                    support_batch = support_batch.to(self.args.device)
                    query_batch = query_batch.to(self.args.device)
                output, y, mask = self.model(support_batch, query_batch)
                loss=self.criterion(output,y,mask)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(epoch_loss)
            self.validation_cnp(valid_tasks, epoch=epoch)
            self.evaluate_cnp(test_tasks, epoch=epoch)

    def validation_cnp(self, valid_tasks,epoch=0):
        self.model.eval()
        i = 0
        predict_all=[]
        target_all=[]
        epoch_loss=0
        for task in valid_tasks:
            support_batch = task.get_support_batch()
            query_batch = task.get_query_batch()
            if self.args.cuda:
                support_batch = support_batch.to(self.args.device)
                query_batch = query_batch.to(self.args.device)
            output, y, mask = self.model(support_batch, query_batch)
            pred = torch.sigmoid(output)
            pred = torch.where(pred > 0.5, 1, 0)
            pred, targets = pred.view(-1,1), y.view(-1,1)
            pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()
            i += 1
            if len(predict_all)==0:
                predict_all=pred
                target_all=targets
            else:
                predict_all=np.vstack((predict_all,pred))
                target_all=np.vstack((target_all,targets))
            loss=self.criterion(output, y, mask)
            epoch_loss += loss
        acc_valid, precision_valid, recall_valid, f1_valid = evaluate_prediction(predict_all, target_all)
        epoch_valid_loss=epoch_loss/len(valid_tasks)
        print('valid loss({},{})'.format(epoch, epoch_valid_loss.item()))
        print("valid Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format( acc_valid, precision_valid, recall_valid, f1_valid))


    def evaluate_cnp(self, test_tasks,epoch=0,print_result=False):
        self.model.eval()
        i = 0
        predict_all=[]
        target_all=[]
        epoch_loss=0
        for task in test_tasks:
            support_batch = task.get_support_batch()
            query_batch = task.get_query_batch()
            if self.args.cuda:
                support_batch = support_batch.to(self.args.device)
                query_batch = query_batch.to(self.args.device)
            output, y, mask = self.model(support_batch, query_batch)
            pred = torch.sigmoid(output)
            pred = torch.where(pred > 0.5, 1, 0)
            pred, targets = pred.view(-1,1), y.view(-1,1)
            pred, targets = pred.cpu().detach().numpy(), targets.cpu().detach().numpy()
            acc, precision, recall, f1 = evaluate_prediction(pred, targets)
            i += 1
            if len(predict_all)==0:
                predict_all=pred
                target_all=targets
            else:
                predict_all=np.vstack((predict_all,pred))
                target_all=np.vstack((target_all,targets))
            loss=self.criterion(output, y, mask)
            epoch_loss += loss
        acc, precision, recall, f1 = evaluate_prediction(predict_all, target_all)
        print("CNP Test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format( acc, precision, recall, f1))
