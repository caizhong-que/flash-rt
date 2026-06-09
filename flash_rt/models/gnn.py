"""GraphSAGE-based GNN for node classification in provenance graphs."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class ProvenanceGNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
