def Get_Adjacent(ids, mapp, edges, hops):
    if hops == 0:
        return set()

    neighbors = set()
    for edge in zip(edges[0], edges[1]):
        if any(mapp[node] in ids for node in edge):
            neighbors.update(mapp[node] for node in edge)

    if hops > 1:
        neighbors = neighbors.union(Get_Adjacent(neighbors, mapp, edges, hops - 1))

    return neighbors

def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

    return prec, rec, fscore, FPR, TPR

def helper(MP, all_pids, GP, edges, mapp):
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    two_hop_gp = Get_Adjacent(GP, mapp, edges, 2)
    two_hop_tp = Get_Adjacent(TP, mapp, edges, 2)

    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    prec, rec, fscore, FPR, TPR = calculate_metrics(TP, FP, FN, TN)
    return prec, rec, fscore

def traverse(ids, mapping, edges, hops, visited=None):
    if hops == 0:
        return set()

    if visited is None:
        visited = set()

    neighbors = set()
    for src, dst in zip(edges[0], edges[1]):
        src_mapped, dst_mapped = mapping[src], mapping[dst]

        if (src_mapped in ids and dst_mapped not in visited) or \
           (dst_mapped in ids and src_mapped not in visited):
            neighbors.add(src_mapped)
            neighbors.add(dst_mapped)

        visited.add(src_mapped)
        visited.add(dst_mapped)

    neighbors.difference_update(ids)
    return ids.union(traverse(neighbors, mapping, edges, hops - 1, visited))

def find_connected_alerts(start_alert, mapping, edges, depth, remaining_alerts):
    connected_path = traverse({start_alert}, mapping, edges, depth)
    return connected_path.intersection(remaining_alerts)

def generate_incident_graphs(alerts, edges, mapping, depth):
    incident_graphs = []
    remaining_alerts = set(alerts)

    while remaining_alerts:
        alert = remaining_alerts.pop()
        connected_alerts = find_connected_alerts(alert, mapping, edges, depth, remaining_alerts)

        if len(connected_alerts) > 1:
            incident_graphs.append(connected_alerts)
            remaining_alerts -= connected_alerts

    return incident_graphs