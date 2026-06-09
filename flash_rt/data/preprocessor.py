"""Graph construction and preprocessing for provenance data."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def build_provenance_graph(
    df: pd.DataFrame, node_type_map: Dict[str, int] = None
) -> Tuple[List[List[str]], List[int], List[List[int]], List[str]]:
    """Build heterogeneous provenance graph from event DataFrame.

    Returns:
        features: Node attribute token lists
        labels: Node type integer labels
        edge_index: COO edge list [[src,...], [dst,...]]
        node_ids: UUIDs for result interpretation
    """
    if node_type_map is None:
        raise ValueError("node_type_map is required")

    required = {"actorID", "actor_type", "objectID", "object_type", "action", "exec", "path"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    nodes: Dict[str, List[str]] = {}
    labels: Dict[str, int] = {}
    edges: List[Tuple[str, str, str]] = []

    for _, row in df.iterrows():
        if row["object_type"] not in node_type_map or row["actor_type"] not in node_type_map:
            continue
        action = row["action"]

        # Actor node
        if row["actorID"] not in nodes:
            nodes[row["actorID"]] = []
            labels[row["actorID"]] = node_type_map[row["actor_type"]]
        _append_attr(nodes[row["actorID"]], row, action)

        # Object node
        if row["objectID"] not in nodes:
            nodes[row["objectID"]] = []
            labels[row["objectID"]] = node_type_map[row["object_type"]]
        _append_attr(nodes[row["objectID"]], row, action)

        edges.append((row["actorID"], row["objectID"], action))

    node_list = list(nodes.keys())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    features = [nodes[nid] for nid in node_list]
    feat_labels = [labels[nid] for nid in node_list]
    edge_index = [[], []]
    for src_id, dst_id, _ in edges:
        edge_index[0].append(node_to_idx[src_id])
        edge_index[1].append(node_to_idx[dst_id])

    return features, feat_labels, edge_index, node_list


def _append_attr(attrs: List[str], row: pd.Series, action: str) -> None:
    attrs.append(str(action))
    if pd.notna(row["path"]) and row["path"] != "":
        attrs.append(str(row["path"]))
    if pd.notna(row["exec"]) and row["exec"] != "":
        attrs.append(str(row["exec"]))


def filter_by_node_types(df: pd.DataFrame, valid_types: set = None) -> pd.DataFrame:
    """Keep only rows whose object_type is in valid_types."""
    if valid_types is None:
        raise ValueError("valid_types is required")
    return df[df["object_type"].isin(valid_types)].reset_index(drop=True)
