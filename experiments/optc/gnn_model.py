import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import device, gnn_weights, gnnTrain
from timer_utils import timer

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(20, 32, normalize=True)
        self.conv2 = SAGEConv(32, 20, normalize=True)
        self.linear = nn.Linear(in_features=20, out_features=4)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with timer("GNN forward total", cuda_sync=True):
            x = self.encode(x, edge_index)
            x = self.linear(x)
            return F.softmax(x, dim=1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with timer("GNN encode: SAGEConv1 + SAGEConv2", cuda_sync=True):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

def build_gnn_model():
    model = GCN().to(device)
    if not gnnTrain:
        with timer(f"load GNN weights: {gnn_weights}", cuda_sync=True):
            model.load_state_dict(torch.load(gnn_weights, map_location=device))
    return model