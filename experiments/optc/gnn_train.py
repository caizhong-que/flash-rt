import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.utils import class_weight

from .config import device, gnn_weights
from .event_processing import load_data
from .timer_utils import timer

def build_training_graph(file_path):
    nodes, labels, edges, mapp, lbl, nemap = load_data(file_path)

    l = np.array(labels)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(l),
        y=l
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, reduction="mean")

    graph = Data(
        x=torch.tensor(nodes, dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device)
    )
    return graph, criterion, mapp, lbl, nemap

def train_model(model, optimizer, criterion, batch):
    model.train()
    optimizer.zero_grad()
    predictions = model(batch.x, batch.edge_index)
    loss = criterion(predictions, batch.y)
    loss.backward()
    optimizer.step()
    return loss.item(), batch.x.size(0)

def evaluate_model_batch(model, batch):
    model.eval()
    with torch.no_grad():
        predictions = model(batch.x, batch.edge_index)
        pred_labels = predictions.argmax(dim=1)
        correct_predictions = int((pred_labels == batch.y).sum())
    return correct_predictions

def run_gnn_training(model, optimizer, criterion, graph, epochs=100, batch_size=5000):
    loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=batch_size)

    for epoch in range(epochs):
        total_loss = total_correct = total_nodes = 0

        with timer(f"GNN epoch {epoch} total", cuda_sync=True):
            for batch in loader:
                loss, nodes = train_model(model, optimizer, criterion, batch)
                total_loss += loss
                total_nodes += nodes
                total_correct += evaluate_model_batch(model, batch)

        average_loss = total_loss / total_nodes
        accuracy = total_correct / total_nodes

        print(f"Epoch #{epoch}. Training Loss: {average_loss:.5f}, Accuracy: {accuracy:.5f}")
        torch.save(model.state_dict(), gnn_weights)
