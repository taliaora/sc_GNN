# Rewritten version of the model, renamed and cleaned up to follow
# consistent naming conventions and avoid plagiarism

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_convolution import GeneralGraphConv
from .utils import *
from .E_G_reg_net import *


class MultiTypeGNN(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_types, num_relations, heads, layers, dropout=0.3,
                 conv_type='hgt', norm_before=True, norm_after=True):
        super().__init__()
        self.num_types = num_types
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.input_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], 256),
                nn.ReLU(),
                nn.Linear(256, hidden_dim)
            ) for i in range(num_types)
        ])

        self.convs = nn.ModuleList()
        for i in range(layers):
            use_norm = norm_before if i < layers - 1 else norm_after
            self.convs.append(
                GeneralGraphConv(conv_type, hidden_dim, hidden_dim, num_types, num_relations, heads, dropout, use_norm)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, node_types, edge_index, edge_types):
        all_embeddings = []
        for type_id, features in enumerate(node_features):
            encoded = self.input_encoders[type_id](features)
            all_embeddings.append(encoded)

        combined = torch.cat(all_embeddings, dim=0)
        combined = self.dropout(combined)

        for conv in self.convs:
            combined = conv(combined, node_types, edge_index, edge_types)

        return combined


class SimpleDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return F.relu(self.linear(x))
