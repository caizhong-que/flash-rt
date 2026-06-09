import json
import pickle
import numpy as np
import torch

from torch_geometric import utils
from torch_geometric.data import Data

from .config import device, xgboost_weights, gnn_weights
from .timer_utils import timer
from .event_processing import Featurize, load_data
from .feature_store import load_gnn_map, load_features
from .rt_train import RT_test
from .graph_utils import helper
from .gnn_model import build_gnn_model


gnn_map = load_gnn_map()

def load_pkl(fname):
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj

def validate(file_path):
    with timer(f"validate total: {file_path}"):
        x, y, _, _ = load_features(file_path)
        xgb_cl = load_pkl(xgboost_weights)

        with timer("validate: xgb.predict"):
            pred = xgb_cl.predict(x)

        with timer("validate: xgb.predict_proba"):
            proba = xgb_cl.predict_proba(x)

        sorted_proba = np.sort(proba, axis=1)
        conf = (sorted_proba[:, -1] - sorted_proba[:, -2]) / sorted_proba[:, -1]
        conf = (conf - conf.min()) / conf.max()

        check = (pred == y)
        flag = ~torch.tensor(check)
        scores = conf[flag].tolist()

    return scores

def load_features_test(dataframe, similarity_threshold=1):
    with timer("load_features_test total"):
        nodes, y_train, edges, mapping, label_map, node_entity_map = Featurize(dataframe)
        X_train = []
        counts = 0

        with timer("load_features_test: build feature matrix"):
            for i, map_id in enumerate(mapping):
                label = label_map[map_id]
                node_embedding = np.zeros(20)

                if label in gnn_map:
                    embedding, stored_set = gnn_map[label]
                    stored_set = set(stored_set)
                    current_set = node_entity_map[map_id]
                    similarity_metric = len(current_set.intersection(stored_set)) / len(current_set.union(stored_set))

                    if similarity_metric >= similarity_threshold:
                        node_embedding = np.array(embedding)
                        counts += 1

                X_train.append(np.hstack((nodes[i], node_embedding)))

    return np.array(X_train), y_train, edges, mapping, counts / len(mapping)

def test():
    with timer("test total", cuda_sync=True):
        file_path = ".\\dataset\\OpTC\\SysClient0051.systemia.com.txt"
        nodes, labels, edges, mapp, lbl, nemap = load_data(file_path)

        graph = Data(
            x=torch.tensor(nodes, dtype=torch.float).to(device),
            y=torch.tensor(labels, dtype=torch.long).to(device),
            edge_index=torch.tensor(edges, dtype=torch.long).to(device)
        )

        model = build_gnn_model()

        with timer("test: load gnn weights", cuda_sync=True):
            model.load_state_dict(torch.load(gnn_weights, map_location=device))

        model.eval()
        with timer("test: GNN encode inference", cuda_sync=True):
            out = model.encode(graph.x, graph.edge_index).tolist()

    return np.hstack([out, out]), labels, edges, mapp

def evaluate_model(df, xgb_cl, similarity_threshold, confidence_threshold):
    with timer("evaluate_model total", cuda_sync=True):
        x, y, edges, mapp, _ = load_features_test(df)

        with timer("evaluate_model: xgb.predict"):
            pred = xgb_cl.predict(x)

        with timer("evaluate_model: xgb.predict_proba"):
            proba = xgb_cl.predict_proba(x)

        sorted_proba = np.sort(proba, axis=1)
        conf = (sorted_proba[:, -1] - sorted_proba[:, -2]) / sorted_proba[:, -1]
        normalized_conf = (conf - conf.min()) / conf.max()

        check = (pred == y) & (normalized_conf > confidence_threshold)
        flag = ~torch.tensor(check)

        confidenceT = conf
        raw_scoreT = 1.0 - confidenceT
        entropyT = -np.sum(proba * np.log(proba + 1e-12), axis=1)

        tripleT = torch.stack([
            torch.from_numpy(raw_scoreT),
            torch.from_numpy(confidenceT),
            torch.from_numpy(entropyT)
        ], dim=1).float()

        window_size = 10
        with timer("evaluate_model: RT_test", cuda_sync=True):
            all_losses = RT_test(tripleT)

        rt_scoreT = torch.zeros(len(tripleT))
        rt_scoreT[:window_size - 1] = 1
        rt_scoreT[window_size - 1:] = all_losses.mean(dim=1)

    return flag, edges, mapp, rt_scoreT

def analyze_events(data_frame, ground_truth_nodes):
    with timer("analyze_events total", cuda_sync=True):
        if data_frame["properties"].apply(lambda x: isinstance(x, str)).any():
            with timer("analyze_events: json.loads(properties)"):
                data_frame["properties"] = data_frame["properties"].apply(json.loads)

        actor_and_object_ids = set(data_frame["actorID"]) | set(data_frame["objectID"])
        relevant_ground_truth = {x for x in ground_truth_nodes if x in actor_and_object_ids}

        with timer("analyze_events: load_features_test"):
            features, labels, edges, mapping, hitrate = load_features_test(data_frame)

        model = load_pkl(xgboost_weights)

        with timer("analyze_events: XGBoost predict"):
            predictions = model.predict(features)

        with timer("analyze_events: XGBoost predict_proba"):
            probabilities = model.predict_proba(features)

        sorted_probabilities = np.sort(probabilities, axis=1)
        confidence_scores = (sorted_probabilities[:, -1] - sorted_probabilities[:, -2]) / sorted_probabilities[:, -1]
        normalized_confidence = (confidence_scores - confidence_scores.min()) / confidence_scores.max()

        misclassified = ~torch.tensor(predictions == labels)

        confidenceT = confidence_scores
        raw_scoreT = 1.0 - confidenceT
        entropyT = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
        tripleT = torch.stack([
            torch.from_numpy(raw_scoreT),
            torch.from_numpy(confidenceT),
            torch.from_numpy(entropyT)
        ], dim=1).float()

        window_size = 10
        with timer("analyze_events: RT_test", cuda_sync=True):
            all_losses = RT_test(tripleT)

        rt_scoreT = torch.zeros(len(tripleT))
        rt_scoreT[:window_size - 1] = 1
        rt_scoreT[window_size - 1:] = all_losses.mean(dim=1)

        rt = rt_scoreT > np.percentile(rt_scoreT, 3)
        rt = torch.logical_and(misclassified, rt)
        index = utils.mask_to_index(rt).tolist()
        ids = set([mapping[x] for x in index])

        with timer("analyze_events: helper metrics"):
            result = helper(set(ids), actor_and_object_ids, relevant_ground_truth, edges, mapping)

    return result, hitrate
