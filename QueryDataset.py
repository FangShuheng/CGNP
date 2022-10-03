from torch_geometric.data import Data, Dataset, DataLoader
import networkx as nx
from multiprocessing import Pool
import random
import torch
import numpy as np
import math

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class CSQueryData(Data):
    def __init__(self, x, edge_index, y, query=None, pos=None, neg=None, raw_feature=None):
        super(CSQueryData, self).__init__()
        self.x = x  # feature
        self.edge_index = edge_index
        self.raw_feature = raw_feature
        self.y = y  # ground truth
        self.query = query
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape) # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0


class CNPQueryData(Data):
    def __init__(self, x, edge_index, y, query=None, pos=None, neg=None, raw_feature=None):
        super(CNPQueryData, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.raw_feature = raw_feature
        self.y = y
        self.query = query
        self.pos = pos
        self.neg = neg
        self.mask = torch.zeros(self.y.shape)  # used as the loss weight
        self.mask[pos] = 1.0
        self.mask[neg] = 1.0

    def __inc__(self, key, value):
        if key == "query":
            return 0
        else:
            return super().__inc__(key, value)


class TaskData(object):
    def __init__(self, all_queries_data, num_shots, seed=4):
        self.all_queries_data = all_queries_data
        self.num_shots = num_shots
        self.seed = seed
        self.support_data, self.query_data = \
            self._support_query_split()
        self.num_support, self.num_query = len(self.support_data), len(self.query_data)

    def _support_query_split(self):
        random.seed(20)
        random.shuffle(self.all_queries_data)
        support_data, query_data = self.all_queries_data[: self.num_shots], self.all_queries_data[self.num_shots:]
        return support_data, query_data

    def get_batch(self):
        loader = DataLoader(self.all_queries_data, batch_size=len(self.all_queries_data), shuffle=False)
        return next(iter(loader))

    def get_support_batch(self, isMaml=False, isSup=False,isRep=False):
        if isMaml:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        elif isSup:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        elif isRep:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        else:
            support_loader = DataLoader(self.support_data, batch_size=self.num_support, shuffle=False)
        return next(iter(support_loader))

    def get_query_batch(self, isMaml=False, isSup=False, isRep=False):
        if isMaml:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        elif isSup:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        elif isRep:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        else:
            query_loader = DataLoader(self.query_data, batch_size=self.num_query, shuffle=False)
        return next(iter(query_loader))


class RawGraphWithCommunity(object):
    def __init__(self, graph, communities, feats):
        self.num_workers = 20
        self.graph = graph  # networkx graph
        self.communities = communities  # list of list
        self.feats = feats  # origin node feat
        self.x_feats = torch.from_numpy(self.feats)
        self.query_index = dict()
        for community in self.communities:
            for node in community:
                if node not in self.query_index:
                    self.query_index[node] = set(community)
                else:
                    self.query_index[node] = self.query_index[node].union(set(community))
        # get the edge index, used by all sampled
        self.edge_index = torch.ones(size=(2, self.graph.number_of_edges()), dtype=torch.long)
        for i, e in enumerate(self.graph.edges()):
            self.edge_index[0][i], self.edge_index[1][i] = e[0], e[1]

    def sample_one_query(self):
        community = random.choice(self.communities)
        query = random.choice(community)
        pos = list(set(community))
        neg = list(set(range(self.graph.number_of_nodes())).difference(set(community)))
        return query, pos, neg

    def sample_one(self, query, num_pos, num_neg):
        pos = list(self.query_index[query])
        neg = list(set(range(self.graph.number_of_nodes())).difference(self.query_index[query]))
        if num_pos<=1:
            masked_pos = random.sample(pos, k= min(math.ceil(num_pos*(len(pos)+len(neg))), len(pos)))
            masked_neg = random.sample(neg, k= min(math.ceil(num_neg*(len(neg)+len(neg))), len(neg)))
            print('graph_node_num:{},num_pos:{},num_neg:{}'.format(self.graph.number_of_nodes(),len(masked_pos),len(masked_neg)))
        else:
            masked_pos = random.sample(pos, k= int(min(num_pos, len(pos))))
            masked_neg = random.sample(neg, k= int(min(num_neg, len(neg))))
        return query, pos, neg, masked_pos, masked_neg

    def get_queries(self, task_size, num_shots):
        if task_size >= len(self.query_index):
            queries = random.sample(list(self.query_index.keys()), k=len(self.query_index))
        else:
            queries = random.sample(list(self.query_index.keys()), k=task_size)
        return queries

    def get_one_query_tensor(self, query, num_pos, num_neg):
        query, pos, neg, masked_pos, masked_neg = self.sample_one(query, num_pos, num_neg)
        x_q = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        x_q[query] = 1
        x, y = self.get_elements_for_query_data(pos, neg, x_q)
        query_data = CSQueryData(x=x, edge_index=self.edge_index, y=y, query=query, pos=masked_pos,
                                 neg=masked_neg, raw_feature=self.x_feats.cpu().numpy())
        return query_data

    def get_one_cnp_query_tenor(self, query,num_pos,num_neg):
        query, pos, neg, masked_pos, masked_neg = self.sample_one(query, num_pos, num_neg)
        x_q = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        x_q[query] = 1
        x_q[masked_pos] = 1
        x_q[masked_neg] = -1
        x, y = self.get_elements_for_query_data(pos, neg, x_q)
        query = torch.LongTensor([query])
        query_data = CNPQueryData(x=x, edge_index=self.edge_index, y=y, query=query, pos=masked_pos,
                                  neg=masked_neg, raw_feature=self.x_feats.cpu().numpy())
        return query_data

    def get_elements_for_query_data(self, pos, neg, x_q):
        x = torch.cat([x_q, self.x_feats], dim=-1)
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        core = nx.core_number(self.graph)
        core_feature = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        for j in range(len(core)):
            core_feature[j] = core[j]
        core_feature = torch.nn.functional.normalize(core_feature, dim=1)
        x = torch.cat([x, core_feature], dim=-1)

        cluster = nx.clustering(self.graph)
        cluster_feature = torch.zeros(size=(self.graph.number_of_nodes(), 1), dtype=torch.float)
        for j in range(len(cluster)):
            cluster_feature[j] = cluster[j]
        x = torch.cat([x, cluster_feature], dim=-1)
        x = x.to(torch.float32)
        y = torch.zeros(size=(self.graph.number_of_nodes(),), dtype=torch.float)
        y = y
        y[pos] = 1
        return x, y


    def get_task(self, queries, num_shots, meta_method, num_pos, num_neg):
        all_queries_data = list()
        if meta_method in ["MAML", "maml", "Transfer", "transfer", "supervised", "ATC", "ATC_noAttr", "ACQ", "ACQ_noAttr", "CTC", "ICSGNN","proto","Reptile","AQDGNN"]:
            for query in queries:
                query_data = self.get_one_query_tensor(query,num_pos,num_neg)
                all_queries_data.append(query_data)
        elif meta_method in ["CNP", "cnp"]:
            for query in queries:
                query_data = self.get_one_cnp_query_tenor(query,num_pos,num_neg)
                all_queries_data.append(query_data)
        else:
            print(f"!!! Unsupported meta_method {meta_method} in RawGraphWithCommunity.get_task")
        task = TaskData(all_queries_data, num_shots=num_shots)
        return task



