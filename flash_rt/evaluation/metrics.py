"""Evaluation metrics for APT detection."""

from typing import Set, List, Dict, Tuple


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, ...]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec, f1, fpr, tpr


def get_adjacent_nodes(node_ids: Set[str], mapping: List[str],
                       edges: List[List[int]], hops: int = 2) -> Set[str]:
    if hops == 0 or not node_ids:
        return set()
    idx_set = {i for i, u in enumerate(mapping) if u in node_ids}
    neighbors = set()
    for s, d in zip(edges[0], edges[1]):
        if s in idx_set or d in idx_set:
            neighbors.add(mapping[s])
            neighbors.add(mapping[d])
    if hops > 1:
        neighbors |= get_adjacent_nodes(neighbors, mapping, edges, hops - 1)
    return neighbors


def evaluate_detection(predicted_ids: Set[str], all_ids: Set[str],
                       gt_malicious: Set[str], edges: List[List[int]],
                       mapping: List[str]) -> Dict[str, float]:
    tp = predicted_ids & gt_malicious
    fp = predicted_ids - gt_malicious
    fn = gt_malicious - predicted_ids
    tn = all_ids - (gt_malicious | predicted_ids)

    # 2-hop relaxation
    two_hop_gt = get_adjacent_nodes(gt_malicious, mapping, edges, 2)
    two_hop_tp = get_adjacent_nodes(tp, mapping, edges, 2)

    tpl = tp | (fn & two_hop_tp)
    fpl = fp - two_hop_gt
    fnl = fn - two_hop_tp

    prec, rec, f1, fpr, tpr = calculate_metrics(len(tpl), len(fpl), len(fnl), len(tn))
    return {"TP": len(tpl), "FP": len(fpl), "FN": len(fnl), "TN": len(tn),
            "precision": prec, "recall": rec, "fscore": f1, "FPR": fpr, "TPR": tpr}
