import torch.nn as nn
import torch
import torch.nn.functional as F
from .FwLayer import LinearFw, MLPFw, get_act_layer
from .FwGNNLayer import GCNConvFw, GATConvFw, SAGEConvFw
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool
from torch.utils.checkpoint import checkpoint

class FwGNN(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(FwGNN, self).__init__()
        self.num_node_feat = node_feat_dim
        self.num_edge_feat = edge_feat_dim
        self.num_layers = args.num_layers
        self.num_hid = args.num_g_hid
        self.num_out = args.gnn_out_dim
        self.model_type = args.gnn_type
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        self.act_type = args.act_type
        self.act_layer = get_act_layer(self.act_type)
        self.gnn_act_layer = get_act_layer(args.gnn_act_type)
        cov_layer = self.build_cov_layer(self.model_type)
        print('network dimension:')
        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid
            print(hidden_input_dim, hidden_output_dim)
            if self.model_type == "GCN" or self.model_type == "GAT" or self.model_type=="SAGE":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            else:
                assert False, "Unsupported model type!"


    def build_cov_layer(self, model_type):
        if model_type == "GCN":
            return lambda in_ch, hid_ch : GCNConvFw(
                in_channels=in_ch, out_channels=hid_ch)
        elif model_type == "GAT":
            return lambda in_ch, hid_ch : GATConvFw(in_channels=in_ch, out_channels=hid_ch)
        elif model_type =="SAGE":
            return lambda in_ch, hid_ch : SAGEConvFw(in_channels=in_ch, out_channels=hid_ch)
        else:
            assert False, "Unsupported model type!"


    def forward(self, x, edge_index, x_batch, edge_attr = None):
        for i in range(self.num_layers):
            if self.model_type == "GCN" or self.model_type == "GAT" or self.model_type == "TGCN" or self.model_type=="SAGE" or self.model_type=="GIN":
                x = self.convs[i](x, edge_index)
            else:
                print("Unsupported model type!")

            if i < self.num_layers - 1:
                if self.act_type != 'relu':
                    x =self.act_layer(x)
                x = F.dropout(x, p = self.dropout, training=self.training)
        return x

class FwCSModel(nn.Module):
    # community search model with fast weight
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(FwCSModel, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.gnn = FwGNN(args, self.node_feat_dim, self.edge_feat_dim)

    def forward(self, batch):
        x, edge_index, x_batch, edge_attr = batch.x, batch.edge_index, batch.batch, batch.edge_attr
        x_hid = self.gnn(x, edge_index, x_batch, edge_attr)
        mask=batch.mask
        return x_hid,mask

