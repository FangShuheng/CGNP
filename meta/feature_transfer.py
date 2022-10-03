import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from util import evaluate_prediction
from torch_geometric.nn import GINConv, GINEConv, NNConv, GATConv, GraphConv, SAGEConv, RGCNConv, TransformerConv
import os
from model.FwLayer import get_act_layer
from model.Loss import WeightBCEWithLogitsLoss
from model.Model import GNN

class Classifier(nn.Module):
    def __init__(self,args, input_dim, gnn_type, act_type, dropout):
        super(Classifier, self).__init__()
        if gnn_type == "GCN":
            self.gnn_layer = GraphConv(in_channels=input_dim, out_channels=1)
        elif gnn_type == "GAT":
            self.gnn_layer = GATConv(in_channels=input_dim, out_channels=1)
        elif gnn_type == "TGCN":
            self.gnn_layer = TransformerConv(in_channels=input_dim, out_channels=1)
        elif gnn_type =="SAGE":
            self.gnn_layer= SAGEConv(in_channels=input_dim, out_channels=1)
        elif gnn_type=="GIN":
            self.gnn_layer =  GINConv(nn.Sequential(nn.Linear(input_dim, 1),
							nn.ReLU(),
							nn.Linear(1, 1)))
        else:
            raise NotImplementedError("Unsupported GNN type!")
        self.dropout = dropout
        self.act_type = act_type
        self.act_layer = get_act_layer(act_type)
        self.gnn = GNN(args, input_dim, 1, args.finetune_layer)

    def forward(self, x, edge_index, x_batch):
        out = self.gnn(x, edge_index, x_batch, x_batch)
        return out

class FeatureTransfer(nn.Module):
    def __init__(self, args, feature_extractor,node_feat):
        super(FeatureTransfer, self).__init__()
        self.args = args
        self.feature_extractor = feature_extractor
        if args.num_layers ==0:
            self.model = Classifier(args,node_feat + 3, args.gnn_type, args.act_type, args.dropout)
        else:
            self.model = Classifier(args,args.gnn_out_dim, args.gnn_type, args.act_type, args.dropout)
        self.criterion = WeightBCEWithLogitsLoss()
        self.update_step = args.update_feattrans
        if self.args.cuda:
            self.feature_extractor.to(self.args.device)
            self.model.to(self.args.device)


    def train_feature_transfer(self, train_tasks, optimizer, scheduler=None):
        train_batches = [task.get_batch() for task in train_tasks]
        self.feature_extractor.train()
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            random.shuffle(train_batches)
            for (i, batch) in enumerate(train_batches):
                if self.args.cuda:
                    batch = batch.to(self.args.device)
                temp, mask= self.feature_extractor(batch)
                output = self.model(temp, batch.edge_index, batch.batch)
                output = output.squeeze()
                loss = self.criterion(output, batch.y, mask)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            if scheduler is not None and self.args.scheduler_type in ["Plateau", "plateau"]:
                scheduler.step(epoch_loss, epoch)
            if scheduler is not None and self.args.scheduler_type in ["Exponential", "exponential"] \
                    and epoch % self.args.decay_patience == 0:
                scheduler.step()
            epoch_loss=epoch_loss/len(train_batches)

    def evaluate_feature_transfer(self, test_tasks, optimizer):
        idx=0
        predict_all=[]
        targets_all=[]
        for i, task in enumerate(test_tasks):
            support_batch = task.get_support_batch()
            query_batch = task.get_query_batch()
            acc, precision, recall, f1, pred, targets = self.evaluate_loop(support_batch, query_batch, optimizer)
            print("Test Task-{}: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(i, acc, precision, recall, f1))
            idx += 1
            if len(predict_all)==0:
                predict_all=pred
                targets_all=targets
            else:
                predict_all = np.vstack((predict_all,pred))
                targets_all = np.vstack((targets_all,targets))
        acc, precision, recall, f1 = evaluate_prediction(predict_all, targets_all)
        print("Feature Transfer Test Result: Acc={:.4f}, Pre={:.4f}, Recall={:.4f}, F1={:.4f}".format(acc, precision, recall, f1))


    def evaluate_loop(self, support_batch, query_batch, optimizer):
        for k in range(0,self.update_step):
            if self.args.cuda:
                support_batch = support_batch.to(self.args.device)
                query_batch = query_batch.to(self.args.device)
            temp,mask=self.feature_extractor(support_batch)
            output= self.model(temp, support_batch.edge_index, support_batch.batch)
            output = output.squeeze()
            loss = self.criterion(output, support_batch.y, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.feature_extractor.eval()
        self.model.eval()
        temp, mask=self.feature_extractor(query_batch)
        query_output= self.model(temp, query_batch.edge_index, query_batch.batch)
        query_output = query_output.squeeze()
        pred = torch.sigmoid(query_output)
        pred = torch.where(pred > 0.5, 1, 0)
        pred, targets = pred.cpu().detach().numpy(), query_batch.y.cpu().detach().numpy()
        acc, precision, recall, f1 = evaluate_prediction(pred, targets)
        pred=pred.reshape(-1,1)
        targets=targets.reshape(-1,1)
        return acc, precision, recall, f1, pred, targets
