from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

def get_args():
    parser = ArgumentParser("MetaCS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

    # Model Settings
    parser.add_argument("--num_layers", default=3, type=int, help="number of gnn conv layers")
    parser.add_argument("--finetune_layer", default=3, type=int, help="number of gnn conv layers for finetune")
    parser.add_argument("--gnn_type", default="GAT", type=str, help="GNN type")
    parser.add_argument("--gnn_act_type", default="relu", type=str,
                        help="activation layer inside gnn aggregate/combine function")
    parser.add_argument("--act_type", default="relu", type=str,
                        help="activation layer function for MLP and between GNN layers")
    parser.add_argument("--num_g_hid", default=128, type=int, help="hidden dim")
    parser.add_argument("--gnn_out_dim", default=128, type=int, help="number of output dimension")
    parser.add_argument("--mlp_hid_dim", default=512, type=int, help="number of hidden units of MLP")
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument("--batch_norm", default=False, type=bool)

    # Meta Learning Setting
    parser.add_argument("--meta_method", default="proto", type=str, help="The meta learning algorithm")  # DKT, CNP, maml
    parser.add_argument("--query_node_num", default=10, type=int)  # total query node number
    parser.add_argument("--num_shots", default=5, type=int)  # support set query node number
    parser.add_argument("--data_set", default='reddit', type=str, help='dataset')  # cora, citeseer, facebook, arxiv, cora_citeseer, reddit, dblp
    parser.add_argument("--data_dir", type=str, default="/home/shfang/data/facebook/facebook")
    parser.add_argument("--subgraph_size", default=200, type=int, help='the size of subgraph sampled in large graph')
    parser.add_argument("--task_num", type=int, help='task number', default=10)
    parser.add_argument("--valid_task_num", type=int, help='valid task number', default=5)
    parser.add_argument("--test_task_num", type=int, default=5, help='the number of test task')
    parser.add_argument("--label_mode", type=str, default='disjoint', help='shared label or disjoint label')
    parser.add_argument("--num_pos", default=0.5, type=float)  # (maximum) proportion of positive instances for each query node
    parser.add_argument("--num_neg", default=0.5, type=float)  # (maximum) proportion of negative instances in each for each query node
    parser.add_argument("--no_feature", default=True, type=bool)

    # maml training configs
    parser.add_argument("--meta_lr", default=1e-3, type=float)  # meta (outer) learning rate of maml #1e-3
    parser.add_argument("--update_step", default=10, type=int)  # inner update step of maml
    parser.add_argument("--update_step_test", default=20, type=int)  # test update step of maml

    # feattrans training configs
    parser.add_argument("--update_feattrans", default=1, type=int)

    #reptile training configs
    parser.add_argument("--epsilon", default=0.1, type=float) #scale: old + scale*(new - old)

    # Training Settings
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)#1e-4-->1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--scheduler_type", default="exponential", type=str, help="the node feature encoding type")  # Exponential/exponential or Plateau/plateau
    parser.add_argument('--decay_factor', type=float, default=0.8, help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10, help='num of epochs for one lr decay.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers for Dataset.')
    parser.add_argument('--seed', default=200, type=int, help='seed')
    parser.add_argument("--pool_type", default="att", type=str, help="CNP Context Pool Type: att, sum, avg")
    parser.add_argument("--facebook_test_indices", default="7-8", type=str,
                        help="indices of facebook tasks used as test set, joined by '-', with the max index as 9, eg."
                             "5-0")
    parser.add_argument("--facebook_valid_indices", default="9-0", type=str,
                        help="indices of facebook tasks used as validation set, joined by '-', with the max index as 9, eg."
                             "3")
    args = parser.parse_args()

    # set the hardware parameter
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    return args
