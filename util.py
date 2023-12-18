import os
import sys
import networkx as nx
import numpy as np
from QueryDataset import RawGraphWithCommunity
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import random
from preprocess_dblp import untar_snap_data

'''evaluate prediction, compute accuracy, precision, recall and F1'''
def evaluate_prediction(pred, targets):
    acc = accuracy_score(targets, pred)
    precision = precision_score(targets, pred)
    recall = recall_score(targets, pred)
    f1 = f1_score(targets, pred)
    return acc, precision, recall, f1

'''
load facebook dataset
load_facebook_featmap: local feature map global feature
load_facebook_graphs: load graphs, communities and feature
'''
def load_facebook_featmap(data_dir: str):
    glob_feat_id = dict()
    f_cnt = 0
    for file_name in os.listdir(data_dir):
        if os.path.splitext(file_name)[1] != '.featnames':
            continue
        with open(os.path.join(data_dir, file_name), "r") as featnames:
            for line in featnames:
                tokens = line.strip().split()
                feat_name = '+'.join(tokens[1:])
                if feat_name not in glob_feat_id.keys():
                    glob_feat_id[feat_name] = f_cnt
                    f_cnt += 1
            featnames.close()
    return glob_feat_id

def load_facebook_graphs(data_dir: str, data_set):
    glob_feat_id = load_facebook_featmap(data_dir)
    num_feat = len(glob_feat_id)
    raw_data_list = list()
    for file_name in os.listdir(data_dir):
        if os.path.splitext(file_name)[1] != '.edges':
            continue
        ego_node_id = int(os.path.splitext(file_name)[0])
        feat_dict = dict()
        with open(os.path.join(data_dir, "{}.featnames".format(ego_node_id)), 'r') as feat_names:
            for line in feat_names:
                tokens = line.strip().split()
                f_id = int(tokens[0])
                feat_name = '+'.join(tokens[1:])
                feat_dict[f_id] = feat_name
            feat_names.close()

        # load input feats
        node_id_dict = dict()
        node_cnt = 0
        with open(os.path.join(data_dir, "{}.feat".format(ego_node_id)), 'r') as feat:
            lines = feat.readlines()
            feats = np.zeros(shape=(len(lines) + 1, len(glob_feat_id)), dtype=np.float)
            for line in lines:
                tokens = line.strip().split()
                node_id_dict[int(tokens[0])] = node_cnt
                for i, val in enumerate(tokens[1:]):
                    if int(val) <= 0:
                        continue
                    idx = glob_feat_id[feat_dict[i]]
                    feats[node_cnt][idx] = 1
                node_cnt += 1
            feat.close()

        # load ego node feature
        with open(os.path.join(data_dir, "{}.egofeat".format(ego_node_id)), 'r') as egofeat:
            node_id_dict[ego_node_id] = node_cnt
            for line in egofeat:
                tokens = line.strip().split()
                for i, val in enumerate(tokens):
                    if int(val) <= 0:
                        continue
                    idx = glob_feat_id[feat_dict[i]]
                    feats[node_cnt][idx] = 1
            egofeat.close()

        # load graph edges
        edge_list = list()
        with open(os.path.join(data_dir, "{}.edges".format(ego_node_id)), "r") as edges:
            for line in edges:
                tokens = line.strip().split()
                src, dst = int(tokens[0]), int(tokens[1])
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
            edges.close()
            ego_edges = [(node_id_dict[ego_node_id], k) for k in node_id_dict.values()]
            edge_list += ego_edges

        # load communities info
        communities = list()
        with open(os.path.join(data_dir, "{}.circles".format(ego_node_id)), 'r') as circles:
            for line in circles:
                tokens = line.strip().split()
                if len(tokens) <= 2:
                    continue
                node_ids = [node_id_dict[int(token)] for token in tokens[1:]]
                communities.append(node_ids)
            circles.close()

        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        print("# of nodes/edges:", graph.number_of_nodes(), graph.number_of_edges())
        raw_data = RawGraphWithCommunity(graph, communities, feats)
        raw_data_list.append(raw_data)
    return raw_data_list, num_feat

def load_dblp_graphs(query_node_num, num_tasks, subgraph_size, label_mode, mode, train_ratio=0.7, valid_ratio=0.1):
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path = "data/com_dblp"
    print(f"Load com_dblp edges")
    if (os.path.exists(path + '//edges.npy') == False):
        untar_snap_data('dblp')
    new_edge = np.load(path + '//edges.npy').tolist()
    graph = nx.from_edgelist(new_edge)
    print(f"Load com_dblp cmty")
    com_list = np.load(path + '//comms.npy', allow_pickle=True).tolist()
    print(len(com_list))
    print(graph.number_of_nodes(), graph.number_of_edges())
    print("------Community List:------")
    node_list_all = range(len(graph))
    raw_data_list = list()
    index = 0

    max_size = subgraph_size \
        if subgraph_size > 0 else np.random.randint(low=100, high=200, size=num_tasks)
    if label_mode == "disjoint":
        if mode=='train':
            com_list = com_list[0:int(train_ratio * len(com_list))]
        elif mode=='valid':
            com_list = com_list[int(train_ratio * len(com_list)):int((train_ratio+valid_ratio) * len(com_list))]
        elif mode=='test':
            com_list = com_list[int((train_ratio+valid_ratio) * len(com_list)):]

    elif label_mode=='shared':
        train_ratio=0.5
        valid_ratio=train_ratio
        if mode=='train':
            #com_list=com_list[0:int(train_ratio * len(com_list))]
            node_list_all=node_list_all[0:int(train_ratio * len(node_list_all))]
        elif mode=='valid':
            #com_list=com_list[int(train_ratio * len(com_list)):int((train_ratio+valid_ratio) * len(com_list))]
            node_list_all=node_list_all[int(train_ratio*len(node_list_all)):int((train_ratio+valid_ratio)*len(node_list_all))]
        elif mode=='test':
            #com_list=com_list[int((train_ratio+valid_ratio) * len(com_list)):]
            node_list_all=node_list_all[int(train_ratio*len(node_list_all)):int((train_ratio+valid_ratio)*len(node_list_all))]

    i=0

    source_ls = random.sample(node_list_all, len(node_list_all))
    while i<num_tasks:  # Each iteration generate one task
        # graph
        label_dict = {}
        sub = nx.Graph()
        sub_new = nx.Graph()
        node_id_dict = dict()
        edge_list = []
        h_hops_neighbor = []
        while len(h_hops_neighbor) < 100:
            node_cnt = 0
            source = source_ls[index]
            index = index + 1
            h_hops_neighbor = []
            h_hops_neighbor.append(source)
            node_id_dict[int(source)] = node_cnt
            pos = 0
            while (pos < len(h_hops_neighbor)) and (pos < max_size) and (
                    len(h_hops_neighbor) < max_size):
                cnode = h_hops_neighbor[pos]
                for nb in graph[cnode]:
                    if (nb not in h_hops_neighbor) and (nb in node_list_all):
                        node_cnt = node_cnt + 1
                        h_hops_neighbor.append(nb)
                        node_id_dict[int(nb)] = node_cnt
                pos = pos + 1
            sub = graph.subgraph(h_hops_neighbor)
            sub_node_list = sub.nodes()
            subedge_list = sub.edges()
            for idx, ege in enumerate(subedge_list):
                src = ege[0]
                dst = ege[1]
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
            sub_new.add_edges_from(edge_list)

        # communities
        sub_com_list = [[] for _ in range(len(com_list))]
        candidate_query_num=0
        for idx, com in enumerate(com_list):
            for node in com:
                if node in sub_node_list:
                    sub_com_list[idx].append(node_id_dict[node])
            candidate_query_num=candidate_query_num+len(sub_com_list[idx])
        while ([] in sub_com_list): sub_com_list.remove([])
        communities = sub_com_list
        if candidate_query_num/3<query_node_num:
            continue
        i=i+1
        print(sub_new.number_of_nodes(), sub_new.number_of_edges())
        print('candidate_query_num:',candidate_query_num)
        feats = np.array([[] for _ in sub_node_list])
        raw_data = RawGraphWithCommunity(sub_new, communities, feats)
        raw_data_list.append(raw_data)
        num_feat = 0
    return raw_data_list, num_feat


'''load data, get tasks and get queries'''
def load_data_and_get_tasks(args):
    query_node_num = args.query_node_num
    num_shots = args.num_shots
    num_pos, num_neg= args.num_pos,args.num_neg
    if args.data_set == 'facebook':
        raw_data_list, node_feat = load_facebook_graphs(args.data_dir,args.data_set)
        queries_list = [raw_data.get_queries(query_node_num, num_shots) for raw_data in raw_data_list]
        tasks = [raw_data.get_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                 zip(raw_data_list, queries_list)]
        test_indices = list(map(int, args.facebook_test_indices.split('-')))
        valid_indices=list(map(int, args.facebook_valid_indices.split('-')))
        test_tasks = [tasks[idx] for idx in test_indices]
        valid_tasks = [tasks[idx] for idx in valid_indices]
        train_tasks = list(set(tasks).difference(set(test_tasks).union(set(valid_tasks))))
        return train_tasks, valid_tasks, test_tasks, node_feat
    elif args.data_set == 'dblp':
        raw_data_list_train, node_feat = load_dblp_graphs(args.query_node_num, args.task_num, args.subgraph_size, args.label_mode,
                                                                       'train')
        raw_data_list_valid, node_feat = load_dblp_graphs(args.query_node_num, args.valid_task_num, args.subgraph_size, args.label_mode,
                                                                      'valid')
        raw_data_list_test, node_feat = load_dblp_graphs(args.query_node_num, args.test_task_num, args.subgraph_size, args.label_mode,
                                                                      'test')

    raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = [], [], [], 0
    queries_list_train = [raw_data.get_queries(query_node_num, num_shots) for raw_data in raw_data_list_train]
    queries_list_valid = [raw_data.get_queries(query_node_num, num_shots) for raw_data in raw_data_list_valid]
    queries_list_test = [raw_data.get_queries(query_node_num, num_shots) for raw_data in raw_data_list_test]

    train_tasks = [raw_data.get_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                   zip(raw_data_list_train, queries_list_train)]
    valid_tasks = [raw_data.get_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                  zip(raw_data_list_valid, queries_list_valid)]
    test_tasks = [raw_data.get_task(queries, num_shots, args.meta_method, num_pos, num_neg) for raw_data, queries in
                  zip(raw_data_list_test, queries_list_test)]
    return train_tasks, valid_tasks, test_tasks, node_feat

