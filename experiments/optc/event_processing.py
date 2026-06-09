import json
import pandas as pd
import numpy as np

from timer_utils import timer
from word2vec_utils import infer

def is_valid_entry(entry):
    valid_objects = {"PROCESS", "FILE", "FLOW", "MODULE"}
    invalid_actions = {"START", "TERMINATE"}

    object_valid = entry["object"] in valid_objects
    action_valid = entry["action"] not in invalid_actions
    actor_object_different = entry["actorID"] != entry["objectID"]

    return object_valid and action_valid and actor_object_different

def Traversal_Rules(data):
    filtered_data = {}

    for entry in data:
        if is_valid_entry(entry):
            key = (
                entry["action"],
                entry["actorID"],
                entry["objectID"],
                entry["object"],
                entry["pid"],
                entry["ppid"]
            )
            filtered_data[key] = entry

    return list(filtered_data.values())

def Sentence_Construction(entry):
    action = entry["action"]
    properties = entry["properties"]
    object_type = entry["object"]

    format_strings = {
        "PROCESS": "{parent_image_path} {action} {image_path} {command_line}",
        "FILE": "{image_path} {action} {file_path}",
        "FLOW": "{image_path} {action} {src_ip} {src_port} {dest_ip} {dest_port} {direction}",
        "MODULE": "{image_path} {action} {module_path}"
    }

    default_format = "{image_path} {action} {module_path}"

    try:
        format_str = format_strings.get(object_type, default_format)
        phrase = format_str.format(action=action, **properties)
    except KeyError:
        phrase = ""

    return phrase.split(" ")

def Extract_Semantic_Info(event):
    object_type = event["object"]
    properties = event["properties"]

    label_mapping = {
        "PROCESS": ("parent_image_path", "image_path"),
        "FILE": ("image_path", "file_path"),
        "MODULE": ("image_path", "module_path"),
        "FLOW": ("image_path", "dest_ip", "dest_port")
    }

    label_keys = label_mapping.get(object_type, None)
    if label_keys:
        labels = [properties.get(key) for key in label_keys]
        if all(labels):
            event["actorname"], event["objectname"] = labels[0], " ".join(labels[1:])
            return event
    return None

def transform(text):
    labeled_data = [event for event in (Extract_Semantic_Info(x) for x in text) if event]
    data = Traversal_Rules(labeled_data)

    phrases = [Sentence_Construction(x) for x in data if Sentence_Construction(x)]
    for datum, phrase in zip(data, phrases):
        datum["phrase"] = phrase

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].str[:-6], format="mixed")
    df.sort_values(by="timestamp", inplace=True)

    return df

def Featurize(df):
    with timer("Featurize total"):
        dummies = {"PROCESS": 0, "FLOW": 1, "FILE": 2, "MODULE": 3}

        nodes = {}
        labels = {}
        lblmap = {}
        neimap = {}
        edges = []

        with timer("Featurize: build nodes/labels/edges"):
            for index, row in df.iterrows():
                actor_id, object_id = row["actorID"], row["objectID"]
                object_type = row["object"]

                nodes.setdefault(actor_id, []).extend(row["phrase"])
                nodes.setdefault(object_id, []).extend(row["phrase"])

                labels[actor_id] = dummies.get("PROCESS", -1)
                labels[object_id] = dummies.get(object_type, -1)

                lblmap[actor_id] = row["actorname"]
                lblmap[object_id] = row["objectname"]

                neimap.setdefault(actor_id, set()).add(row["objectname"])
                neimap.setdefault(object_id, set()).add(row["actorname"])

                edge_type = row["properties"]["direction"] if object_type == "FLOW" else row["action"]
                edges.append((actor_id, object_id, edge_type))

        features, feat_labels, edge_index = [], [], [[], []]
        node_index = {}

        with timer("Featurize: Word2Vec semantic infer for nodes"):
            for node, phrases in nodes.items():
                if not (len(phrases) == 1 and phrases[0] == "DELETE"):
                    features.append(infer(phrases))
                    feat_labels.append(labels[node])
                    node_index[node] = len(features) - 1

        with timer("Featurize: build edge_index"):
            for src, dst, _ in edges:
                edge_index[0].append(node_index[src])
                edge_index[1].append(node_index[dst])

        mapp = list(node_index.keys())

    return features, np.array(feat_labels), edge_index, mapp, lblmap, neimap

def load_data(file_path):
    with timer(f"load_data total: {file_path}"):
        with timer("load_data: read json lines"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = [json.loads(line) for line in file]

        df = transform(content)
        result = Featurize(df)

    return result