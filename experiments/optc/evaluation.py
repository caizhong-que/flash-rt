"""Evaluate attack scenarios on OpTC data with GNN+XGBoost+Transformer."""

import numpy as np
import pandas as pd
import torch
from torch_geometric import utils

from .config import XGBOOST_WEIGHTS, DATA_DIR
from .stream_loader import load_events_from_hosts, load_ground_truth
from .event_processing import transform
from .evaluation_utils import load_pkl, evaluate_model


def evaluate_attack(host: str, attack_name: str, similarity_threshold: float,
                    confidence_threshold: float, rt_percentile: float = 99,
                    rt_logic: str = "and"):
    all_events = load_events_from_hosts([host])
    entity_set = {e["actorID"] for e in all_events} | {e["objectID"] for e in all_events}
    df = transform(all_events)
    gt = {x for x in load_ground_truth(DATA_DIR / "optc.txt") if x in entity_set}

    xgb = load_pkl(XGBOOST_WEIGHTS)
    flag, edges, mapping, rt_score = evaluate_model(df, xgb, similarity_threshold, confidence_threshold)
    pred_ids = {mapping[i] for i in utils.mask_to_index(flag).tolist()}

    row = {"Host": host, "Attack": attack_name, "Events": len(all_events),
           "Entities": len(entity_set), "GT": len(gt),
           "Base_Pred": len(pred_ids)}

    from flash_rt.evaluation.metrics import evaluate_detection  # reuse
    base = evaluate_detection(pred_ids, entity_set, gt, edges, mapping)
    row.update({f"Base_{k}": v for k, v in base.items() if k in ("precision", "recall", "fscore")})

    if rt_percentile is not None:
        th = np.percentile(rt_score.cpu().numpy(), rt_percentile)
        rt_mask = rt_score > th
        final = torch.logical_and(flag, rt_mask) if rt_logic == "and" else torch.logical_or(flag, rt_mask)
        rt_ids = {mapping[i] for i in utils.mask_to_index(final).tolist()}
        rt = evaluate_detection(rt_ids, entity_set, gt, edges, mapping)
        row.update({"RT_Pred": len(rt_ids)})
        row.update({f"RT_{k}": v for k, v in rt.items() if k in ("precision", "recall", "fscore")})

    return row


def main():
    attacks = [
        {"host": "051", "attack_name": "Malicious Upgrade Attack",
         "similarity_threshold": 1, "confidence_threshold": 0.6},
        {"host": "201", "attack_name": "Plain PowerShell Empire Attack",
         "similarity_threshold": 1, "confidence_threshold": 0.5},
        {"host": "501", "attack_name": "Custom PowerShell Empire Attack",
         "similarity_threshold": 1, "confidence_threshold": 0.98, "rt_logic": "or"},
    ]
    rows = [evaluate_attack(**cfg) for cfg in attacks]
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    result_df.to_csv("optc_attack_results.csv", index=False)


if __name__ == "__main__":
    main()
