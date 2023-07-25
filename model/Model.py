import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Layer import MLP, FC, NNGINConv, NNGINConcatConv
from model.FwLayer import get_act_layer
from torch_scatter import scatter_mean
from torch_geometric.nn import GINConv, GINEConv, NNConv, GATConv, GraphConv, SAGEConv, RGCNConv, TransformerConv
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool


class GNN(nn.Module):
    def __init__(self, args, node_feat_dim, out_dim, num_layer):
        super(GNN, self).__init__()
        self.num_node_feat = node_feat_dim
        self.num_layers = num_layer
        self.num_hid = args.num_g_hid
        self.num_out = out_dim
        self.model_type = args.gnn_type
        self.dropout = args.dropout
        self.convs = nn.ModuleList()
        self.act_type = args.act_type
        self.act_layer = get_act_layer(self.act_type)
        self.gnn_act_layer = get_act_layer(args.gnn_act_type)
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

            if self.model_type == "GAT"  or self.model_type == "GCN" or self.model_type == "SAGE":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            else:
                assert False, "Unsupported model type!"


    def build_cov_layer(self, model_type):
        if model_type == "GAT":
            return GATConv
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "GCN":
            return GraphConv
        else:
            assert False, "Unsupported model type!"

    def forward(self, x, edge_index, x_batch, edge_attr = None):

        for i in range(self.num_layers):
            if self.model_type == "GAT" or self.model_type =="GCN" or self.model_type == "SAGE":
                x = self.convs[i](x, edge_index)
            else:
                print("Unsupported model type!")
            if i < self.num_layers - 1:
                if self.act_type != 'relu':
                    x = self.act_layer(x)
                x = F.dropout(x, p = self.dropout, training=self.training)
        return x


# community search model
class CSCNP(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(CSCNP, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.query_size = args.query_node_num - args.num_shots
        self.num_shots = args.num_shots
        self.gnn = GNN(args, self.node_feat_dim, args.gnn_out_dim, args.num_layers)

    def forward(self, support_batch, query_batch):
        context = self.encode(support_batch)
        hid, y, mask = self.decode(context, query_batch)
        return hid, y, mask

    def encode(self, support_batch):
        x, edge_index, x_batch, edge_attr = \
            support_batch.x, support_batch.edge_index, support_batch.batch, support_batch.edge_attr
        x_hid = self.gnn(x, edge_index, x_batch, edge_attr)
        x_hid = x_hid.view(self.num_shots, -1, self.gnn.num_out)
        context = torch.sum(x_hid, dim=0, keepdim=False)
        return context

    def decode(self, context, query_batch):
        query, y, mask = query_batch.query, query_batch.y, query_batch.mask
        q = context[query]
        hid = torch.einsum("nc,kc->nk", [q, context])
        hid=torch.flatten(hid)
        return hid, y, mask


class CNPEncoder(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim):
        super(CNPEncoder, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.query_size = args.query_node_num - args.num_shots
        self.num_shots = args.num_shots
        self.pool_type = args.pool_type
        self.gnn = GNN(args, self.node_feat_dim, args.gnn_out_dim,args.num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=self.gnn.num_out, num_heads=1)
        self.LK = nn.Linear(self.gnn.num_out, self.gnn.num_out)
        self.LV = nn.Linear(self.gnn.num_out, self.gnn.num_out)
        self.LQ = nn.Linear(self.gnn.num_out, self.gnn.num_out)

    def forward(self, support_batch):
        x, edge_index, x_batch, edge_attr = \
            support_batch.x, support_batch.edge_index, support_batch.batch, support_batch.edge_attr
        x_hid = self.gnn(x, edge_index, x_batch, edge_attr)
        x_hid = x_hid.view(self.num_shots, -1, self.gnn.num_out)
        if self.pool_type in ["SUM", "sum"]:
            context = torch.sum(x_hid, dim=0, keepdim=False)  # [num_node, num_out]
        elif self.pool_type in ["AVG", "avg"]:
            context = torch.mean(x_hid, dim=0, keepdim=False)  # [num_node, num_out]
        elif self.pool_type in ["Att", "att"]:
            Q = self.LK(x_hid) # [support_size, num_node, num_out]
            V = self.LV(x_hid) # [support_size, num_node, num_out]
            K = self.LK(x_hid) # [support_size, num_node, num_out]
            attn_output, _ = self.attention(Q, K, V) # [support_size, num_node, num_out]
            context = torch.sum(attn_output, dim=0, keepdim=False)  # [num_node, num_out]
        else:
            raise NotImplementedError("Unsupported Context Pooling type!")
        return context

class CNPGNNDecoder(nn.Module):
    def __init__(self, args):
        super(CNPGNNDecoder, self).__init__()
        self.args = args
        self.args.num_layers = 2
        self.query_size = args.query_node_num - args.num_shots
        self.num_shots=args.num_shots
        self.gnn = GNN(self.args, self.args.gnn_out_dim, args.gnn_out_dim,args.num_layers)

    def forward(self, context, support_edge_index, support_x_batch, query_batch):
        support_edge_index = support_edge_index.view(2, self.num_shots, -1)[:, 0, :]
        support_x_batch = support_x_batch.view(self.num_shots, -1)[0]
        context = self.gnn(context, support_edge_index, support_x_batch)
        query, y, mask = query_batch.query, query_batch.y, query_batch.mask
        q = context[query]  # q : [num_shots, num_out]
        hid = torch.einsum("nc,kc->nk", [q, context])  # [num_shots, num_node]
        hid=torch.flatten(hid)
        return hid, y, mask

class CNPMLPDecoder(nn.Module):
    def __init__(self, args, hid_ch, out_ch):
        super(CNPMLPDecoder, self).__init__()
        self.query_size = args.query_node_num - args.num_shots
        self.mlp = MLP(in_ch=args.gnn_out_dim, hid_ch=hid_ch, out_ch=out_ch)

    def forward(self, context, support_edge_index, support_x_batch, query_batch):
        context = self.mlp(context) # [num_node, gnn_out_dim] --> [num_node, self.mlp.out_ch]
        query, y, mask = query_batch.query, query_batch.y, query_batch.mask
        q = context[query]  # q : [num_shots, num_out]
        hid = torch.einsum("nc,kc->nk", [q, context])  # [num_shots, num_node]
        hid = torch.flatten(hid)
        return hid, y, mask


class CSCNPComp(nn.Module):
    def __init__(self, args, node_feat_dim, edge_feat_dim, decoder_type):
        super(CSCNPComp, self).__init__()
        self.encoder = CNPEncoder(args, node_feat_dim, edge_feat_dim)
        if decoder_type == "GNN":
            self.decoder = CNPGNNDecoder(args)
        elif decoder_type == "MLP":
            self.decoder = CNPMLPDecoder(args, hid_ch=512, out_ch=512)
        else:
            raise NotImplementedError("Unsupported CNP Decoder type!")

    def forward(self, support_batch, query_batch):
        context = self.encoder(support_batch)
        hid, y, mask = self.decoder(context, support_batch.edge_index, support_batch.batch, query_batch)
        return hid, y, mask



