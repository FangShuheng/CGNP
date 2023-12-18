import networkx as nx
import numpy as np
import random
import os, sys
import gzip
import pathlib
import  tarfile
import os.path as osp

def main():
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)
    path="data/com_dblp"
    print(f"Load com_dblp edges")
    if(os.path.exists(path + '//edges.npy') == False):
        untar_snap_data('dblp')
    new_edge=np.load(path+'//edges.npy').tolist()
    graph = nx.from_edgelist(new_edge)
    print(f"Load com_dblp cmty")
    com_list=np.load(path+'//comms.npy',allow_pickle=True).tolist()
    print("------Community List:------")
    print(f"# of nodes: {graph.number_of_nodes()}, # of edges: {graph.number_of_edges()}")
    print([len(com) for com in com_list if len(com) >= 100])
    for com in com_list: print(f"min: {min(com)}, max: {max(com)}, size: {len(com)}")


def untar_snap_data(name):
    """Load the snap comm datasets."""
    print(f'Untar {name} edge')
    root = pathlib.Path('XXX/com_dblp')
    with open(root / f'com-{name}.ungraph.txt', 'rt') as fh:
        edges = fh.read().strip().split('\n')[4:]
    edges = [[int(i) for i in e.split()] for e in edges]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    print(f'Untar {name} cmty')
    with open(root / f'com-{name}.top5000.cmty.txt', 'rt') as fh:
        comms = fh.readlines()
    comms = [[mapping[int(i)] for i in x.split()] for x in comms]
    root = pathlib.Path()/'data'/f'com_{name}'
    root.mkdir(exist_ok=True, parents=True)
    np.save(root/'edges',edges)
    np.save(root/'comms',comms,allow_pickle=True)
    np.save(root/'map',mapping,allow_pickle=True)

if __name__ == "__main__":
    main()