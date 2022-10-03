import os
import networkx as nx
from networkx.classes.function import subgraph
from networkx.generators.sudoku import sudoku_graph
import numpy as np
import random
import torch
import pickle as pkl
import sys
import dgl
import csv
import queue
import json
from torch.utils.data import dataset
from QueryDataset import RawGraphWithCommunity
from torch_geometric.datasets import Reddit
from torch_geometric.utils import convert
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score

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


'''load cora/citeseer dataset'''
def load_citation_graphs(query_node_num, data_set, num_tasks, subgraph_size, label_mode, mode):
    if data_set == 'cora':
        with open("/home/shfang/MetaCS/code_comsearch/Data/cora/ind.cora.graph",'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
    if data_set == 'citeseer':
        with open("/home/shfang/MetaCS/code_comsearch/Data/citeseer/ind.citeseer.graph",'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
    edge_list = []
    for i in range(len(graph)):
        for j, node2 in enumerate(graph[i]):
            edge_list.append((i, node2))
    g = nx.Graph()
    g.add_edges_from(edge_list)

    if label_mode == 'disjoint':
        if mode == 'train':
            node_list_all = []
            label_list=[]
            with open('/home/shfang/MetaCS/data/disjoint/' + data_set + '/train.csv') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i, row in enumerate(csvreader):
                    filename = row[1]
                    label=row[2]
                    if label not in label_list:
                        label_list.append(label)
                    n_idx = int(filename.split('_')[1])
                    node_list_all.append(n_idx)
        elif mode == 'val':
            node_list_all = []
            label_list=[]
            with open('/home/shfang/MetaCS/data/disjoint/' + data_set + '/val.csv') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i, row in enumerate(csvreader):
                    filename = row[1]
                    label=row[2]
                    if label not in label_list:
                        label_list.append(label)
                    n_idx = int(filename.split('_')[1])
                    node_list_all.append(n_idx)
        elif mode == 'test':
            node_list_all = []
            label_list=[]
            with open('/home/shfang/MetaCS/data/disjoint/' + data_set + '/test.csv') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i, row in enumerate(csvreader):
                    filename = row[1]
                    label=row[2]
                    if label not in label_list:
                        label_list.append(label)
                    n_idx = int(filename.split('_')[1])
                    node_list_all.append(n_idx)
    elif label_mode == 'shared':
        if mode == 'train':
            node_list_all = []
            label_list=[]
            with open('/home/shfang/MetaCS/data/shared/' + data_set + '/train.csv') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i, row in enumerate(csvreader):
                    filename = row[1]
                    label=row[2]
                    if label not in label_list:
                        label_list.append(label)
                    n_idx = int(filename.split('_')[1])
                    node_list_all.append(n_idx)
        elif mode == 'val':
            node_list_all = []
            label_list=[]
            with open('/home/shfang/MetaCS/data/shared/' + data_set + '/val.csv') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i, row in enumerate(csvreader):
                    filename = row[1]
                    label=row[2]
                    if label not in label_list:
                        label_list.append(label)
                    n_idx = int(filename.split('_')[1])
                    node_list_all.append(n_idx)
        elif mode == 'test':
            node_list_all = []
            label_list=[]
            with open('/home/shfang/MetaCS/data/shared/' + data_set + '/test.csv') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i, row in enumerate(csvreader):
                    filename = row[1]
                    label=row[2]
                    if label not in label_list:
                        label_list.append(label)
                    n_idx = int(filename.split('_')[1])
                    node_list_all.append(n_idx)

    feature = np.load('/home/shfang/MetaCS/MAML_comsearch/data/' + data_set + '/features.npy', allow_pickle=True)

    num_feat = feature.shape[1]
    with open('/home/shfang/MetaCS/MAML_comsearch/data/' + data_set + '/label.pkl', 'rb') as f:
        info = pkl.load(f)

    raw_data_list = list()
    source_ls = random.sample(node_list_all, len(node_list_all))
    index = 0
    i=0
    max_size = subgraph_size\
        if subgraph_size > 0 else np.random.randint(low=100, high=200, size=num_tasks)

    while i< num_tasks:
        # graph
        label_dict = {}
        sub = nx.Graph()
        sub_new = nx.Graph()
        node_id_dict = dict()
        edge_list = []
        h_hops_neighbor = []
        while len(h_hops_neighbor) < subgraph_size/2:
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
                    if (nb not in h_hops_neighbor):
                        node_cnt = node_cnt + 1
                        h_hops_neighbor.append(nb)
                        node_id_dict[int(nb)] = node_cnt
                pos = pos + 1
            sub = g.subgraph(h_hops_neighbor)
            node_list = sub.nodes()
            subedge_list = sub.edges()
            for idx, ege in enumerate(subedge_list):
                src = ege[0]
                dst = ege[1]
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
            sub_new.add_edges_from(edge_list)

        # communities
        communities = list()
        label_dict = {}
        candidate_query_num=0
        for idx, node in enumerate(node_list):
            if  (node in node_list_all):
                if info[str('0' + '_' + str(node))][0] in label_dict.keys():
                    label_dict[info[str('0' + '_' + str(node))][0]].append(node_id_dict[node])
                else:
                    label_dict[info[str('0' + '_' + str(node))][0]] = [node_id_dict[node]]
        for idx, node_ls in enumerate(label_dict):
            communities.append(label_dict[node_ls])
            candidate_query_num=candidate_query_num+len(label_dict[node_ls])
        #ensure each graph has enough queries
        if candidate_query_num < query_node_num:
            continue
        i=i+1
        print(sub_new.number_of_nodes(), sub_new.number_of_edges())
        # feats
        feats = np.vstack(([feature[np.array(x)] for j, x in enumerate(node_list)]))
        raw_data = RawGraphWithCommunity(sub_new, communities, feats)
        raw_data_list.append(raw_data)

    return raw_data_list, num_feat

'''load cora2citeseer or citeseer2cora dataset'''
def load_transfer_citation_graphs(data_set, num_tasks, subgraph_size, mode):
    if data_set == 'cora' or data_set == 'citeseer':
        with open("/home/shfang/MetaCS/code_comsearch/Data/{0}/ind.{0}.graph".format(data_set),
                  'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
    edge_list = []
    for i in range(len(graph)):
        for j, node2 in enumerate(graph[i]):
            edge_list.append((i, node2))
    g = nx.Graph()
    g.add_edges_from(edge_list)

    if mode == 'train':
        node_list_all = []
        with open('/home/shfang/MetaCS/data/cora_citeseer/' + data_set + '/train.csv') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i, row in enumerate(csvreader):
                filename = row[1]
                n_idx = int(filename.split('_')[1])
                node_list_all.append(n_idx)
    elif mode == 'test':
        node_list_all = []
        with open('/home/shfang/MetaCS/data/cora_citeseer/' + data_set + '/test.csv') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i, row in enumerate(csvreader):
                filename = row[1]
                n_idx = int(filename.split('_')[1])
                node_list_all.append(n_idx)
    elif mode == 'val':
        node_list_all = []
        with open('/home/shfang/MetaCS/data/cora_citeseer/' + data_set + '/val.csv') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i, row in enumerate(csvreader):
                filename = row[1]
                n_idx = int(filename.split('_')[1])
                node_list_all.append(n_idx)

    feature = np.load('/home/shfang/MetaCS/data/cora_citeseer/' + data_set + '/features.npy', allow_pickle=True)
    num_feat = feature.shape[1]
    with open('/home/shfang/MetaCS/data/cora_citeseer/' + data_set + '/label.pkl', 'rb') as f:
        info = pkl.load(f)

    raw_data_list = list()
    source_ls = random.sample(node_list_all, len(node_list_all))
    index = 0

    max_size = subgraph_size \
        if subgraph_size > 0 else np.random.randint(low=200, high=250, size=num_tasks)
    for i in range(num_tasks):
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
            sub = g.subgraph(h_hops_neighbor)
            node_list = sub.nodes()
            subedge_list = sub.edges()
            for idx, ege in enumerate(subedge_list):
                src = ege[0]
                dst = ege[1]
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
            sub_new.add_edges_from(edge_list)
        print(sub_new.number_of_nodes(), sub_new.number_of_edges())

        # communities
        communities = list()
        for idx, node in enumerate(node_list):
            if info[str('0' + '_' + str(node))][0] in label_dict.keys():
                label_dict[info[str('0' + '_' + str(node))][0]].append(node_id_dict[node])
            else:
                label_dict[info[str('0' + '_' + str(node))][0]] = [node_id_dict[node]]
        for idx, node_ls in enumerate(label_dict):
            communities.append(label_dict[node_ls])

        # feats
        feats = np.vstack(([feature[np.array(x)] for j, x in enumerate(node_list)]))

        raw_data = RawGraphWithCommunity(sub_new, communities, feats)
        raw_data_list.append(raw_data)

    return raw_data_list, num_feat

'''load arxiv dataset'''
def load_arxiv_graphs(no_feature, query_node_num, data_set, num_tasks, subgraph_size, label_mode, mode):
    with open('/home/shfang/MetaCS/MAML_comsearch/data/' + data_set + '/label.pkl', 'rb') as f:
        info = pkl.load(f)
    with open('/home/shfang/MetaCS/MAML_comsearch/data/' + data_set + '/graph_dgl.pkl', 'rb') as f:
        graph = pkl.load(f)
        graph = graph[0].to_networkx()
    if label_mode == 'disjoint':
        node_list_all = []
        label_list=[]
        with open(os.path.join('/home/shfang/MetaCS/data/disjoint/arxiv/', mode + '.csv')) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i, row in enumerate(csvreader):
                filename = row[1]
                label=row[2]
                if label not in label_list:
                    label_list.append(label)
                n_idx = int(filename.split('_')[1])
                node_list_all.append(n_idx)
        print('labellist:', label_list)
    elif label_mode == 'shared':
        node_list_all = range(len(graph))

    raw_data_list = list()
    source_ls = random.sample(node_list_all, len(node_list_all))
    index = 0
    max_size = subgraph_size\
        if subgraph_size > 0 else np.random.randint(low=100, high=200, size=num_tasks)

    #construct subgraph
    i=0
    while i<num_tasks:
        sub = nx.Graph()
        sub_new = nx.Graph()
        node_id_dict = dict()
        edge_list = []
        h_hops_neighbor = []
        while len(h_hops_neighbor) < 100:
            node_cnt = 0
            pos = 0
            source = source_ls[index]
            index = index + 1
            h_hops_neighbor = []
            h_hops_neighbor.append(source)
            node_id_dict[int(source)] = node_cnt
            while (pos < len(h_hops_neighbor)) and (pos < max_size) and (len(h_hops_neighbor) < max_size):
                cnode = h_hops_neighbor[pos]
                for nb in graph[cnode]:
                    if (nb not in h_hops_neighbor):
                        node_cnt = node_cnt + 1
                        h_hops_neighbor.append(nb)
                        node_id_dict[int(nb)] = node_cnt
                pos = pos + 1
            sub = graph.subgraph(h_hops_neighbor)
            node_list = sub.nodes()
            subedge_list = sub.edges()
            for idx, ege in enumerate(subedge_list):
                src = ege[0]
                dst = ege[1]
                edge_list.append((node_id_dict[src], node_id_dict[dst]))
            sub_new.add_edges_from(edge_list)

        # communities
        communities = list()
        label_dict = {}
        candidate_query_num=0
        for idx, node in enumerate(node_list):
            if  (node in node_list_all):
                if info[str('0' + '_' + str(node))] in label_dict.keys():
                    label_dict[info[str('0' + '_' + str(node))]].append(node_id_dict[node])
                else:
                    label_dict[info[str('0' + '_' + str(node))]] = [node_id_dict[node]]
        for idx, node_ls in enumerate(label_dict):
            communities.append(label_dict[node_ls])
            candidate_query_num=candidate_query_num+len(label_dict[node_ls])
        #ensure each graph has enough queries
        if candidate_query_num < query_node_num:
            continue
        i=i+1
        print(sub_new.number_of_nodes(), sub_new.number_of_edges())
        # feats
        if no_feature==False:
            feature = np.load('/home/shfang/MAML_comsearch/data/' + data_set + '/features.npy', allow_pickle=True)
            feats = np.vstack(([feature[np.array(x)] for j, x in enumerate(node_list)]))
            num_feat = feature.shape[1]
        else:
            feats = np.array([[] for _ in node_list])
            num_feat=0
        raw_data = RawGraphWithCommunity(sub_new, communities, feats)
        raw_data_list.append(raw_data)
    return raw_data_list, num_feat

'''
load reddit dataset
sample_reddit_bfs: sample subgraph
load_reddit: load graphs, communities and feature
'''
def sample_reddit_bfs(query_id, max_size, candidate, graph, data, glob_communities):
    node_list = []
    Q = queue.Queue()
    Q.put(query_id)
    seen = [query_id]
    while not Q.empty():
        cur = Q.get()
        node_list.append(cur)
        if len(node_list) >= max_size:
            break
        for nb in graph[cur]:
            if nb in seen:
                continue
            Q.put(nb)
            seen.append(nb)
    # reorder the nodes
    node_id_dict = {l: n_id for n_id, l in enumerate(node_list)}#key original id; value new id;
    edge_list = graph.subgraph(node_list).edges()
    edge_list = [(node_id_dict[src], node_id_dict[dst]) for (src, dst) in edge_list]

    communities = list()
    candidate_query_number=0
    temp_list=set(node_list).intersection(set(candidate))
    for k, val in glob_communities.items():#key:label value:node id
        temp_comm = set(val).intersection(temp_list)  # get the local community induced by node_list
        temp_comm = [node_id_dict[node] for node in temp_comm]
        if len(temp_comm) <= 2:
            continue
        communities.append(temp_comm)
        candidate_query_number=candidate_query_number+len(temp_comm)
    res_graph = nx.Graph()
    res_graph.add_edges_from(edge_list)
    feats = data.x[node_list].numpy()
    print("subgraph of {} size: {}, {}".format(query_id, res_graph.number_of_nodes(), res_graph.number_of_edges()))
    return res_graph, communities, feats, candidate_query_number

def load_reddit(no_feature, query_node_num, num_tasks, valid_task_num ,test_task_num, label_mode: str, subgraph_size, train_ratio=0.7, valid_ratio=0.1):
    path = os.path.join("./", "reddit")
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = Reddit(path)
    data = dataset[0]
    num_feat = 0 if no_feature else data.x.shape[1]

    print("Reddit loaded!!!")

    edge_list = data.edge_index.t().tolist()
    graph = nx.Graph()
    graph.add_nodes_from(list(range(data.num_nodes)))
    graph.add_edges_from(edge_list)

    #import communities
    print(graph.number_of_nodes(), graph.number_of_edges())
    glob_communities = dict()
    for node_id, label in enumerate(data.y.numpy().tolist()):
        if label not in glob_communities.keys():
            glob_communities[label] = list()
        glob_communities[label].append(node_id)

    node_candidate = list(range(graph.number_of_nodes()))

    #split into train/val/test
    if label_mode == "disjoint":
        train_valid_community_idx = random.sample(range(len(glob_communities)), k=int((train_ratio+valid_ratio) * len(glob_communities)))
        train_community_idx=train_valid_community_idx[0:int(train_ratio*len(glob_communities))]
        valid_community_idx=train_valid_community_idx[int(train_ratio*len(glob_communities)):]
        train_node_candidate = list()
        valid_node_candidate = list()
        test_node_candidate = list()
        for i, label in enumerate(glob_communities.keys()):
            if i in train_community_idx:
                train_node_candidate += glob_communities[label]
            elif i in valid_community_idx:
                valid_node_candidate += glob_communities[label]
            else:
                test_node_candidate += glob_communities[label]
    elif label_mode == "shared":
        train_node_candidate = random.sample(range(len(node_candidate)), k=int(train_ratio * len(node_candidate)))
        valid_test_node_candidate = list(set(node_candidate).difference(set(train_node_candidate)))
        valid_node_candidate=random.sample(range(len(node_candidate)), k=int((1-train_ratio)/2 * len(node_candidate)))
        test_node_candidate=list(set(valid_test_node_candidate).difference(set(valid_node_candidate)))
    else:
        raise NotImplementedError("Unsupported label mode")

    #construct training tasks
    raw_data_list = list()
    queries = list()
    print("generating raw_data_list_train...")
    query_id_ls = random.sample(train_node_candidate, len(train_node_candidate))
    idx = 0
    while len(raw_data_list) < num_tasks:
        query_id = query_id_ls[idx]
        idx = idx + 1
        if query_id in queries:
            continue
        queries.append(query_id)

        raw_data_name = f"reddit_{subgraph_size}_{label_mode}_subgraph_of_{query_id}.npy"\
            if subgraph_size > 0 and subgraph_size != 100 else f"reddit_{label_mode}_subgraph_of_{query_id}.npy"
        raw_data_path = os.path.join('/home/shfang/MetaCS/', './saved_query', raw_data_name)

        if os.path.exists(raw_data_path):
            print(f"subgraph of {query_id} found, loading from file")
            raw_data = np.load(raw_data_path, allow_pickle=True)[0]
        else:
            max_size = subgraph_size\
                if subgraph_size > 0 else random.randint(100, 200)

            res_graph, communities, feats, candidate_query_number\
                = sample_reddit_bfs(query_id, max_size, train_node_candidate, graph, data, glob_communities)
            raw_data = RawGraphWithCommunity(res_graph, communities, feats)
        if no_feature:
            raw_data.feats = np.array([[] for _ in raw_data.graph.nodes()])
            raw_data.x_feats = torch.from_numpy(raw_data.feats)
        if raw_data.graph.number_of_nodes() >= 100 and len(raw_data.query_index) >= query_node_num:
            np_save_if_not_existed(raw_data_path, raw_data)
            raw_data_list.append(raw_data)
        else:
            print("raw_data.res_graph.number_of_nodes() < 100 or (candidate_query_number < query_node_num)")

    #construct valid tasks
    raw_data_list_valid=list()
    queries=list()
    print("generating raw_data_list_validation...")
    query_id_ls = random.sample(valid_node_candidate, len(valid_node_candidate))
    idx = 0
    while len(raw_data_list_valid) < valid_task_num:
        query_id = query_id_ls[idx]
        idx = idx + 1
        if query_id in queries:
            continue
        queries.append(query_id)
        raw_data_name = f"reddit_{subgraph_size}_{label_mode}_subgraph_of_{query_id}.npy"\
            if subgraph_size > 0 and subgraph_size != 100 else f"reddit_{label_mode}_subgraph_of_{query_id}.npy"
        raw_data_path = os.path.join('/home/shfang/MetaCS/', './saved_query', raw_data_name)
        if os.path.exists(raw_data_path):
            print(f"subgraph of {query_id} found, loading from file")
            raw_data = np.load(raw_data_path, allow_pickle=True)[0]
        else:
            max_size = subgraph_size\
                if subgraph_size > 0 else random.randint(100, 200)
            res_graph, communities, feats, candidate_query_number\
                = sample_reddit_bfs(query_id, max_size, valid_node_candidate, graph, data, glob_communities)
            raw_data = RawGraphWithCommunity(res_graph, communities, feats)
        if no_feature:
            raw_data.feats = np.array([[] for _ in raw_data.graph.nodes()])
            raw_data.x_feats = torch.from_numpy(raw_data.feats)
        if raw_data.graph.number_of_nodes() >= 100 and len(raw_data.query_index) >= query_node_num:
            np_save_if_not_existed(raw_data_path, raw_data)
            raw_data_list_valid.append(raw_data)
        else:
            print("raw_data.res_graph.number_of_nodes() < 100 or (candidate_query_number < query_node_num)")

    #construct test tasks
    raw_data_list_test = list()
    queries = list()
    print("generating raw_data_list_test...")
    query_id_ls = random.sample(test_node_candidate, len(test_node_candidate))
    idx = 0
    while len(raw_data_list_test) < test_task_num:
        query_id = query_id_ls[idx]
        idx = idx + 1
        if query_id in queries:
            continue
        queries.append(query_id)
        raw_data_valid_name = f"reddit_{subgraph_size}_{label_mode}_valid-subgraph_of_{query_id}.npy"\
            if subgraph_size > 0 and subgraph_size != 100 else f"reddit_{label_mode}_valid-subgraph_of_{query_id}.npy"
        raw_data_path = os.path.join('/home/shfang/MetaCS/', './saved_query', raw_data_valid_name)
        if os.path.exists(raw_data_path):
            print(f"valid subgraph of {query_id} found, loading from file")
            raw_data = np.load(raw_data_path, allow_pickle=True)[0]
        else:
            max_size = subgraph_size\
                if subgraph_size > 0 else random.randint(100, 200)
            res_graph, communities, feats, candidate_query_number\
                = sample_reddit_bfs(query_id, max_size, test_node_candidate, graph, data, glob_communities)
            raw_data = RawGraphWithCommunity(res_graph, communities, feats)
        if no_feature:
            raw_data.feats = np.array([[] for _ in raw_data.graph.nodes()])
            raw_data.x_feats = torch.from_numpy(raw_data.feats)
        if raw_data.graph.number_of_nodes() >= 100 and len(raw_data.query_index) >= query_node_num:
            np_save_if_not_existed(raw_data_path, raw_data)
            raw_data_list_test.append(raw_data)
        else:
            print("raw_data.res_graph.number_of_nodes() < 100 or (candidate_query_number < query_node_num)")
    return raw_data_list, raw_data_list_valid, raw_data_list_test, num_feat

def np_save_if_not_existed(path, saved_data):
    if not os.path.exists(path):
        saved_data_numpy = np.array([saved_data], dtype=object)
        np.save(path, saved_data_numpy)

'''load dblp dataset'''
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
        valid_ratio=train_ratio
        if mode=='train':
            node_list_all=node_list_all[0:int(train_ratio * len(node_list_all))]
        elif mode=='valid':
            node_list_all=node_list_all[int(train_ratio*len(node_list_all)):int((train_ratio+valid_ratio)*len(node_list_all))]
        elif mode=='test':
            node_list_all=node_list_all[int(train_ratio*len(node_list_all)):int((train_ratio+valid_ratio)*len(node_list_all))]

    i=0
    source_ls = random.sample(node_list_all, len(node_list_all))
    #Each iteration generate one task
    while i<num_tasks:
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

    raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = [], [], [], 0
    if args.data_set == 'cora' or args.data_set == 'citeseer':
        raw_data_list_train, node_feat = load_citation_graphs(args.query_node_num,args.data_set, args.task_num, args.subgraph_size,
                                                              args.label_mode, 'train')
        raw_data_list_valid, node_feat = load_citation_graphs(args.query_node_num,args.data_set, args.valid_task_num, args.subgraph_size,
                                                             args.label_mode, 'val')
        raw_data_list_test, node_feat = load_citation_graphs(args.query_node_num,args.data_set, args.test_task_num, args.subgraph_size,
                                                             args.label_mode, 'test')
    elif args.data_set == 'arxiv':
        raw_data_list_train, node_feat = load_arxiv_graphs( args.no_feature, args.query_node_num, args.data_set, args.task_num, args.subgraph_size,
                                                           args.label_mode, 'train')
        raw_data_list_valid, node_feat = load_arxiv_graphs(args.no_feature, args.query_node_num, args.data_set, args.valid_task_num, args.subgraph_size,
                                                            args.label_mode, 'val')
        raw_data_list_test, node_feat = load_arxiv_graphs(args.no_feature, args.query_node_num, args.data_set, args.test_task_num, args.subgraph_size,
                                                          args.label_mode, 'test')
    elif args.data_set == 'reddit':
        raw_data_list_train, raw_data_list_valid, raw_data_list_test, node_feat = \
            load_reddit(no_feature=args.no_feature, query_node_num=args.query_node_num, num_tasks=args.task_num, valid_task_num=args.valid_task_num, test_task_num=args.test_task_num, label_mode=args.label_mode, subgraph_size=args.subgraph_size)
    elif args.data_set == 'cora_citeseer':
        raw_data_list_train, node_feat = load_transfer_citation_graphs('cora', args.task_num, args.subgraph_size,
                                                                       'train')
        raw_data_list_valid, node_feat = load_transfer_citation_graphs('citeseer', args.valid_task_num, args.subgraph_size,
                                                                       'val')
        raw_data_list_test, node_feat = load_transfer_citation_graphs('citeseer', args.test_task_num,
                                                                      args.subgraph_size, 'test')
    elif args.data_set == 'citeseer_cora':
        raw_data_list_train, node_feat = load_transfer_citation_graphs('citeseer', args.task_num, args.subgraph_size,
                                                                       'train')
        raw_data_list_valid, node_feat = load_transfer_citation_graphs('cora', args.valid_task_num, args.subgraph_size,
                                                                       'val')
        raw_data_list_test, node_feat = load_transfer_citation_graphs('cora', args.test_task_num, args.subgraph_size,
                                                                      'test')
    elif args.data_set == 'dblp':
        raw_data_list_train, node_feat = load_dblp_graphs(args.query_node_num, args.task_num, args.subgraph_size, args.label_mode,
                                                                       'train')
        raw_data_list_valid, node_feat = load_dblp_graphs(args.query_node_num, args.valid_task_num, args.subgraph_size, args.label_mode,
                                                                      'valid')
        raw_data_list_test, node_feat = load_dblp_graphs(args.query_node_num, args.test_task_num, args.subgraph_size, args.label_mode,
                                                                      'test')

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

