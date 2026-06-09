import json
import pandas as pd

from event_processing import transform

def read_event_data(host):
    file_path = f".\\data\\OpTC\\SysClient0{host}.systemia.com.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]

def load_events_from_hosts(hosts):
    all_events = []
    for host in hosts:
        path = f".\\data\\OpTC\\SysClient0{host}.systemia.com.txt"
        with open(path, "r", encoding="utf-8") as file:
            raw_events = [json.loads(line) for line in file]
        all_events.extend(raw_events)
    return all_events

def load_ground_truth(gt_file):
    with open(gt_file, "r") as file:
        gt_nodes = set(file.read().split())
    return gt_nodes

def stream_events(batch_size, window_size):
    event_buffer = {}
    hosts = ["051"]
    positions = {host: 0 for host in hosts}

    while True:
        for host in hosts:
            if host not in event_buffer or len(event_buffer[host]) < positions[host] + batch_size:
                events = read_event_data(host)
                dframe = transform(events)
                if host in event_buffer:
                    event_buffer[host] = pd.concat([event_buffer[host], dframe], ignore_index=True)
                else:
                    event_buffer[host] = dframe

            start = positions[host]
            end = start + batch_size
            yield event_buffer[host][start:end]

            positions[host] += window_size
            if positions[host] >= len(event_buffer[host]):
                return