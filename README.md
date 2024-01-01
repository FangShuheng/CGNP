Community Search: A Meta-Learning Approach
-----------------
A PyTorch + torch-geometric implementation of CGNP, as described in the paper: Shuheng Fang, Kangfei Zhao, Guanghua Li, Jeffery Xu Yu. [Community Search: A Meta-Learning Approach]


### Requirements
```
python 3.8
networkx
numpy
scipy
scikit-learn
torch 1.7.1
torch-geometric 1.7
```

Import the conda environment by running
```
conda env create -f cgnp.yaml
conda activate cgnp
pip install -r requirements.txt
```


### Quick Start
Running Facebook
```
python main.py    \
       --data_set facebook     \
       --gnn_type GAT     \
       --meta_method cnp      \
       --data_dir [your/own/directory/containing/facebook/dataset (i.e. /home/shfang/data/facebook/facebook)]  \
       --pool_type avg
```


### Key Parameters
All the parameters with their default value are in get_args.py

| name | type   | description |
| ----- | --------- | ----------- |
| num_layers  | int    | number of GNN layers    |
| gnn_type | string |  type of GNN layer (GCN, GAT, SAGE)     |
| pool_type  | string | type of the Commutative Operation (att, mean,sum)  |
| meta_method | string | type of different meta learning algorithm |
| epochs  | int   | number of training epochs  |
| query_node_num  | int   | total number of query nodes for one task  |
| num_shots  | int   | number of query nodes of support set for one task|
| subgraph_size  | int   | size of subgraph sampled in large graph |
| data_set  | string   | dataset |
| task_num  | int   | number of training tasks |
| test_task_num  | int   | number of testing tasks |
| label_mode  | string   | community mode: shared community or disjoint community |
| num_pos  | float   | maximum proportion of positive instances for each query node |
| num_neg  | float   | maximum proportion of negative instances for each query node |
| learning_rate | Float   | learning rate  |


### Project Structure
```
main.py         # project extrance
util.py         # generate tasks for different dataset, evaluation
QueryDataset.py # extract query from subgraphs and generate support/query set for meta algorithm
get_args.py     # parameters settings

    /meta
        cnp.py                       # train and test for CGNP
    /model
        FwLayer.py
        Model.py                      # model for CGNP
        Layer.py                      # GNN layers and other layers
        Loss.py
```
To use your own dataset, you can put the data graphs, feature, ground truth communities to
'/data/disjoint(shared)/DATASET_NAME/graph_dgl.pkl', '/data/disjoint(shared)/DATASET_NAME/features.npy', '/data/disjoint(shared)/DATASET_NAME/label.pkl', respectively. And to preprocess, divide the graph into two parts, train.csv and test.csv, you can refer to citation_preprocess.py.

The format of input graph Arxiv/Cora/Citeseer and feature follows [G-Meta](https://github.com/mims-harvard/G-Meta) and you can also get Arxiv dataset from it;
The Reddit/Cora/Citeseer datasets are from [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html);
For DBLP, you can download it in [SNAP] (https://snap.stanford.edu/data/com-DBLP.html);
For Facebook, find it in [SNAP] (https://snap.stanford.edu/data/ego-Facebook.html).


### Contact
Open an issue or send email to shfang@se.cuhk.edu.hk if you have any problem

### Cite Us
```
@inproceedings{fang2023community,
  title={Community search: a meta-learning approach},
  author={Fang, Shuheng and Zhao, Kangfei and Li, Guanghua and Yu, Jeffrey Xu},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={2358--2371},
  year={2023},
  organization={IEEE}
}
```
