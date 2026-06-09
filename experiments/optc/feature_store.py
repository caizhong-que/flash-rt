import json
import numpy as np
import torch

from config import emb_store_path
from event_processing import load_data
from timer_utils import timer

def create_embedding_store(model, graph, mapp, lbl, nemap, output_path=emb_store_path):
    model.eval()

    with torch.no_grad():
        out = model.encode(graph.x, graph.edge_index).tolist()

    gnn_map = {}
    for i in range(len(mapp)):
        gnn_map[lbl[mapp[i]]] = (out[i], list(nemap[mapp[i]]))

    with open(output_path, "w") as file:
        json.dump(gnn_map, file)

    return gnn_map

def load_gnn_map(path=emb_store_path):
    with open(path, "r") as file:
        gnn_map = json.load(file)
    return gnn_map

def load_features(filename=None, similarity=1, gnn_map=None):
    with timer(f"load_features total: {filename}"):
        if gnn_map is None:
            gnn_map = load_gnn_map()

        nodes, y_train, edges, mapp, lbl, nemap = load_data(filename)
        zero_vector = np.zeros(20)

        X_train = []
        with timer("load_features: concat Word2Vec feature + GNN store feature"):
            for idx, map_item in enumerate(mapp):
                label = lbl[map_item]
                node_feature = nodes[idx]

                if label in gnn_map:
                    emb, stored_set = gnn_map[label]
                    stored_set = set(stored_set)
                    current_set = nemap[map_item]
                    jaccard_similarity = len(current_set.intersection(stored_set)) / len(current_set.union(stored_set))
                    feature_vector = emb if jaccard_similarity >= similarity else zero_vector
                else:
                    feature_vector = zero_vector

                X_train.append(np.hstack((node_feature, feature_vector)))

    return np.array(X_train), y_train, edges, mapp