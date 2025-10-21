#heterograph construction 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from torchmetrics.functional import pairwise_cosine_similarity

class GeneralGraphConv(nn.Module):
    def __init__(self, conv_type, input_dim, output_dim, num_node_types, num_relations, heads, dropout, use_norm=True):
        super().__init__()
        self.conv_type = conv_type
        self.attention_weights = None

        if conv_type == 'hgt':
            self.layer = HeteroGraphConv(input_dim, output_dim, num_node_types, num_relations, heads, dropout, use_norm)
        elif conv_type == 'gcn':
            self.layer = GCNConv(input_dim, output_dim)
        elif conv_type == 'gat':
            self.layer = GATConv(input_dim, output_dim // heads, heads=heads)

    def forward(self, x, node_types, edge_index, edge_types):
        if self.conv_type == 'hgt':
            out = self.layer(x, node_types, edge_index, edge_types)
            self.attention_weights = self.layer.attention_weights
            return out
        return self.layer(x, edge_index)

class HeteroGraphConv(MessagePassing):
    def __init__(self, input_dim, output_dim, num_node_types, num_relations, heads, dropout=0.2, use_norm=True, init_type='uniform', **kwargs):
        super().__init__(node_dim=0, aggr='add', **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = output_dim // heads
        self.scale = math.sqrt(self.head_dim)
        self.num_node_types = num_node_types
        self.num_relations = num_relations
        self.use_norm = use_norm
        self.init_type = init_type

        # Define per-node-type transformation layers
        self.k_linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_node_types)])
        self.q_linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_node_types)])
        self.v_linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_node_types)])
        self.a_linears = nn.ModuleList([nn.Linear(output_dim, output_dim) for _ in range(num_node_types)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(output_dim) for _ in range(num_node_types)]) if use_norm else None

        # Relation-specific parameters
        self.rel_pri = nn.Parameter(torch.ones(num_relations, heads))
        self.rel_att = nn.Parameter(torch.Tensor(num_relations, heads, self.head_dim, self.head_dim))
        self.rel_msg = nn.Parameter(torch.Tensor(num_relations, heads, self.head_dim, self.head_dim))

        self.skip_weights = nn.Parameter(torch.ones(num_node_types))
        self.dropout = nn.Dropout(dropout)

        # Initialize relation matrices
        glorot(self.rel_att)
        glorot(self.rel_msg)

        self.attention_weights = None
        self.updated_output = None

    def forward(self, x, node_types, edge_index, edge_types):
        return self.propagate(edge_index, x=x, node_types=node_types, edge_types=edge_types)

    def message(self, edge_index_i, x_i, x_j, node_types_i, node_types_j, edge_types):
        num_edges = edge_index_i.size(0)
        device = x_i.device

        self.attention_weights = torch.zeros(num_edges, self.heads, device=device)
        messages = torch.zeros(num_edges, self.heads, self.head_dim, device=device)

        for src_type in range(self.num_node_types):
            mask_src = node_types_j == src_type
            k_transform = self.k_linears[src_type]
            v_transform = self.v_linears[src_type]

            for tgt_type in range(self.num_node_types):
                mask_tgt = (node_types_i == tgt_type) & mask_src
                q_transform = self.q_linears[tgt_type]

                for rel_type in range(self.num_relations):
                    rel_mask = (edge_types == rel_type) & mask_tgt
                    if rel_mask.sum() == 0:
                        continue

                    q = q_transform(x_i[rel_mask]).view(-1, self.heads, self.head_dim)
                    k = k_transform(x_j[rel_mask]).view(-1, self.heads, self.head_dim)
                    v = v_transform(x_j[rel_mask]).view(-1, self.heads, self.head_dim)

                    k = torch.bmm(k.transpose(1, 0), self.rel_att[rel_type]).transpose(1, 0)
                    v = torch.bmm(v.transpose(1, 0), self.rel_msg[rel_type]).transpose(1, 0)

                    att_score = (q * k).sum(dim=-1) * self.rel_pri[rel_type] / self.scale
                    self.attention_weights[rel_mask] = att_score
                    messages[rel_mask] = v

        att_weights = softmax(self.attention_weights, edge_index_i).unsqueeze(-1)
        return (messages * att_weights).view(-1, self.output_dim)

    def update(self, aggregated, x, node_types):
        output = F.gelu(aggregated)
        device = x.device
        result = torch.zeros_like(output)

        for t in range(self.num_node_types):
            mask = node_types == t
            if mask.sum() == 0:
                continue

            transformed = self.dropout(self.a_linears[t](output[mask]))
            skip_factor = torch.sigmoid(self.skip_weights[t])

            if self.use_norm:
                result[mask] = self.norm_layers[t](transformed * skip_factor + x[mask] * (1 - skip_factor))
            else:
                result[mask] = transformed * skip_factor + x[mask] * (1 - skip_factor)

        self.updated_output = result
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, node_types={self.num_node_types}, relations={self.num_relations})"
