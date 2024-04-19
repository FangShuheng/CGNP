import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
import pickle
import dgl
from tqdm import tqdm
import json
from texttable import Texttable
from scipy.sparse import coo_matrix
import sys
import pickle as pkl
import scipy.sparse as sp
from networkx.readwrite import json_graph
import json
import os


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def graph_reader(dataset_str):
    names = ['graph']
    objects = []
    for i in range(len(names)):
        with open("/.../Data/{0}/ind.{0}.{1}".format(dataset_str, names[i]), 'rb') as f: #
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))


    graph=objects[0]
    edge_list=[]
    for i in range(len(graph)):
        for j,node2 in enumerate(graph[i]):
            edge_list.append((i,node2))
    adj=nx.Graph()
    adj.add_edges_from(edge_list)
    print(adj.number_of_nodes(),adj.number_of_edges())
    return adj

def feature_reader(dataset_str, compression=0):

    names = ['x',  'tx',  'allx']
    objects = []
    for i in range(len(names)):
        with open("/.../Data/{0}/ind.{0}.{1}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, tx, allx  = tuple(objects)
    test_idx_reorder = parse_index_file("/.../Data/{0}/ind.{0}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.tocoo()
    features = features.toarray()

    feature_list = []
    feature_list.append(features)

    return features

def target_reader(dataset_str):

    names = [ 'y',  'ty',  'ally']
    objects = []
    for i in range(len(names)):
        with open("/.../Data/{0}/ind.{0}.{1}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty,  ally = tuple(objects)
    test_idx_reorder = parse_index_file("/.../Data/{0}/ind.{0}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)


    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    ally = np.argmax(ally, axis=1)
    ty = np.argmax(ty, axis=1)
    labels = np.concatenate((ally, ty))
    labels[test_idx_reorder] = labels[test_idx_range]
    labels = labels.reshape(-1,1)
    return labels

def main():
    dataset='citeseer'
    mode='shared_label'
    path=os.path.join('/.../shared/',dataset)
    graph=graph_reader(dataset)
    dgl_Gs=[dgl.DGLGraph(graph)]
    feature_map = feature_reader(dataset)
    label_map = target_reader(dataset)

    # G is a dgl graph
    info = {}
    for j in range(len(label_map)):
        info['0' + '_' + str(j)] = label_map[j]

    if mode=='disjoint_label':
        if dataset=='cora':
            num_of_labels=7
            num_label_set=3
        if dataset=='citeseer':
            num_of_labels=6
            num_label_set=3
        if dataset=='pubmed':
            num_of_labels=3
            num_label_set=3

        df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})

        labels = np.unique(list(range(num_of_labels)))

        test_labels = np.random.choice(labels, num_label_set, False)
        labels_left = [i for i in labels if i not in test_labels]
        val_labels = np.random.choice(labels_left, num_label_set, False)
        train_labels = [i for i in labels_left]

        df[df.label.isin(train_labels)].reset_index(drop = True).to_csv(path + '/train.csv')
        df[df.label.isin(val_labels)].reset_index(drop = True).to_csv(path + '/val.csv')
        df[df.label.isin(test_labels)].reset_index(drop = True).to_csv(path + '/test.csv')

    elif mode=='shared_label':
        if dataset=='cora':
            num_of_labels=7
        if dataset=='citeseer':
            num_of_labels=6
        if dataset=='pubmed':
            num_of_labels=3
        df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})
        labels = np.unique(list(range(num_of_labels)))

        test_labels = labels
        val_labels = labels
        train_labels = labels

        df[df.label.isin(train_labels)].reset_index(drop = True).to_csv(path + '/train.csv')
        df[df.label.isin(val_labels)].reset_index(drop = True).to_csv(path + '/val.csv')
        df[df.label.isin(test_labels)].reset_index(drop = True).to_csv(path + '/test.csv')




    with open(path +'/graph_dgl.pkl', 'wb') as f:
        pickle.dump(dgl_Gs, f)

    with open(path + '/label.pkl', 'wb') as f:
        pickle.dump(info, f)

    np.save(path + '/features.npy', np.array(feature_map))



if __name__=='__main__':
    main()
