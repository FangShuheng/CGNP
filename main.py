from util import load_data_and_get_tasks
from meta.cnp import CNP
from model.Model import CSCNP, CSCNPComp
import numpy as np
import torch
import time
import random
from get_args import get_args



def main(args):
    node_feat, train_tasks, valid_tasks, test_tasks = 0,[], [], []
    print('-----getting tasks------')
    if args.meta_method in ["CNP", "cnp"]:
        train_tasks, valid_tasks, test_tasks, node_feat = load_data_and_get_tasks(args)
    print('-----start training------')
    if args.meta_method in ["CNP", "cnp"]:
        run_all_cnps(args, node_feat, train_tasks, valid_tasks, test_tasks)


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

