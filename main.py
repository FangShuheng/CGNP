from meta.reptile import Reptile
from util import load_data_and_get_tasks
from util import evaluate_prediction
from meta.maml import MAML
from meta.cnp import CNP
from meta.feature_transfer import FeatureTransfer
from meta.reptile import Reptile
from model.FwModel import FwCSModel
from model.Model import CSModel, CSCNP, CSCNPComp, protoCS
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import time
import random
import os
import networkx as nx
from get_args import get_args
from model.Loss import WeightBCEWithLogitsLoss
import threading
import pylab as pl
from torch_geometric.utils import to_dense_adj


def main(args):
    node_feat, train_tasks, valid_tasks, test_tasks = 0,[], [], []
    print('-----getting tasks------')
    if args.meta_method in ["MAML", "maml", "CNP", "cnp", "Transfer", "transfer","Reptile"]:
        train_tasks, valid_tasks, test_tasks, node_feat = load_data_and_get_tasks(args)
    print('-----start training------')
    if args.meta_method in ["MAML", "maml"]:
        run_maml(args, node_feat, train_tasks, valid_tasks, test_tasks)
    elif args.meta_method in ["CNP", "cnp"]:
        run_all_cnps(args, node_feat, train_tasks, valid_tasks, test_tasks)
    elif args.meta_method in ["Transfer", "transfer"]:
        run_transfer(args, node_feat, train_tasks, valid_tasks, test_tasks)
    elif args.meta_method in ["Reptile"]:
        run_reptile(args, node_feat, train_tasks, valid_tasks, test_tasks)
    else:
        assert False, "Unsupported meta learning algorithm!"

'''Feature Transfer'''
def run_transfer(args, node_feat, train_tasks, valid_tasks, test_tasks):
    print('-----Linear Transfer-----')
    args.num_layers = args.num_layers-args.finetune_layer
    feature_extractor = CSModel(args, node_feat_dim=node_feat + 3, edge_feat_dim=10)
    feature_transfer = FeatureTransfer(args, feature_extractor,node_feat)
    train_optimizer = optim.Adam([{'params': feature_transfer.model.parameters(), 'lr': args.learning_rate},
                                  {'params': feature_transfer.feature_extractor.parameters(),
                                   'lr': args.learning_rate}], weight_decay=args.weight_decay)
    fine_tune_optimizer = optim.Adam([{'params': feature_transfer.parameters(), 'lr': 0.001}])
    t_start = time.time()
    feature_transfer.train_feature_transfer(train_tasks, train_optimizer)
    t_middle = time.time()
    feature_transfer.evaluate_feature_transfer(test_tasks, fine_tune_optimizer)
    t_end = time.time()
    print('train time={:.4f}, test time={:.4f}'.format(t_middle - t_start, t_end - t_middle))

'''MAML'''
def run_maml(args, node_feat, train_tasks, valid_tasks, test_tasks):
    print('-----MAML-----')
    args.gnn_out_dim = 1
    model = FwCSModel(args, node_feat_dim=node_feat + 3, edge_feat_dim=1)  # unused edge feat dim
    maml = MAML(args, model)
    t_start = time.time()
    maml.train_MAML(train_tasks)
    t_middle = time.time()
    maml.evaluate_maml(test_tasks)
    t_end = time.time()
    print('train time={:.4f}, test time={:.4f}'.format(t_middle - t_start, t_end - t_middle))

'''Reptile'''
def run_reptile(args, node_feat, train_tasks, valid_tasks, test_tasks):
    print('-----Reptile-----')
    args.gnn_out_dim = 1
    model = CSModel(args, node_feat_dim=node_feat + 3, edge_feat_dim=10)
    reptile = Reptile(args, model)
    t_start = time.time()
    reptile.train_Reptile(train_tasks)
    t_middle = time.time()
    reptile.evaluate_Reptile(test_tasks)
    t_end = time.time()
    print('train time={:.4f}, test time={:.4f}'.format(t_middle - t_start, t_end - t_middle))

'''
our method: CGNP
There are three decoder types: inner product; MLP; GNN
'''
def run_cnp_model(args, node_feat, train_tasks, valid_tasks, test_tasks, wandb, decoder_str=""):
    if decoder_str in ["MLP", "GNN"]:
        model = CSCNPComp(args, node_feat_dim=node_feat + 3, edge_feat_dim=10, decoder_type=decoder_str)
    else:
        model = CSCNP(args, node_feat_dim=node_feat + 3, edge_feat_dim=10)
    print('model:\n', model)
    cnp = CNP(args, model,wandb)
    t_start = time.time()
    cnp.train_cnp(train_tasks, valid_tasks, test_tasks)
    t_middle = time.time()
    cnp.evaluate_cnp(test_tasks, print_result=True)
    t_end = time.time()
    print('train time={:.4f}, test time={:.4f}'.format(t_middle - t_start, t_end - t_middle))


def run_all_cnps(args, node_feat, train_tasks, valid_tasks, test_tasks, wandb_run):
    print('-----CGNP-----')
    args.gnn_out_dim = 128
    #three types of CGNP: inner product;MLP;GNN
    for decoder_str in ["", "MLP", "GNN"]:
        run_cnp_model(args, node_feat, train_tasks, valid_tasks,test_tasks, wandb_run, decoder_str)


if __name__ == "__main__":
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    main(args)

