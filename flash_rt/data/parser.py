"""CDM (Common Data Model) provenance log parser for DARPA E3 datasets."""

import csv
import gzip
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import orjson

INCLUDE_NODE_TYPES = {"MemoryObject", "NetFlowObject", "UnnamedPipeObject"}
CDM_NAMESPACE = "com.bbn.tc.schema.avro.cdm18"
EVENT_TYPE = f"{CDM_NAMESPACE}.Event"
UUID_KEY = f"{CDM_NAMESPACE}.UUID"

_EXCLUDED_RECORD_TYPES = {
    b"com.bbn.tc.schema.avro.cdm18.Event",
    b"com.bbn.tc.schema.avro.cdm18.Host",
    b"com.bbn.tc.schema.avro.cdm18.TimeMarker",
    b"com.bbn.tc.schema.avro.cdm18.StartMarker",
    b"com.bbn.tc.schema.avro.cdm18.UnitDependency",
    b"com.bbn.tc.schema.avro.cdm18.EndMarker",
}
_EXCLUDED_RE = re.compile(b"|".join(re.escape(t) for t in _EXCLUDED_RECORD_TYPES))


@dataclass(slots=True)
class ProvenanceEdge:
    actor_id: str
    actor_type: str
    object_id: str
    object_type: str
    action: str
    timestamp: str
    exec_path: str = ""
    file_path: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"actorID": self.actor_id, "actor_type": self.actor_type,
                "objectID": self.object_id, "object_type": self.object_type,
                "action": self.action, "timestamp": self.timestamp,
                "exec": self.exec_path, "path": self.file_path}


class CDMParser:
    """Two-pass parser for DARPA CDM provenance logs (Avro/JSON in tar.gz)."""

    EDGE_FIELDS = ["actorID", "actor_type", "objectID", "object_type",
                   "action", "timestamp", "exec", "path"]

    def __init__(self, include_nodes: Optional[Set[str]] = None):
        self.include_nodes = include_nodes or INCLUDE_NODE_TYPES
        self._node_type_map: Dict[str, str] = {}

    def parse_archive(self, tar_path: Path, output_dir: Path,
                      chunk_size: int = 100000) -> Tuple[int, int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._node_type_map = self._collect_nodes(tar_path)
        num_edges = self._extract_edges(tar_path, output_dir, chunk_size)
        return len(self._node_type_map), num_edges

    def _collect_nodes(self, tar_path: Path) -> Dict[str, str]:
        node_map: Dict[str, str] = {}
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                with tar.extractfile(member) as f:
                    for line in f:
                        if not line.strip() or _EXCLUDED_RE.search(line):
                            continue
                        try:
                            obj = orjson.loads(line)
                        except orjson.JSONDecodeError:
                            continue
                        uuid, nt = self._extract_node_type(obj)
                        if uuid and nt:
                            node_map[uuid] = nt
        return node_map

    def _extract_node_type(self, obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        datum = obj.get("datum", {})
        if not datum:
            return None, None
        try:
            key, value = next(iter(datum.items()))
        except StopIteration:
            return None, None
        if not isinstance(value, dict):
            return None, None
        uuid = value.get("uuid")
        if not uuid:
            return None, None
        node_class = key.split(".")[-1]
        node_type = value.get("type", "")
        if not node_type and node_class in self.include_nodes:
            node_type = node_class
        return (uuid, node_type) if node_type else (None, None)

    def _extract_edges(self, tar_path: Path, output_dir: Path, chunk_size: int) -> int:
        total = 0
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                edges: List[ProvenanceEdge] = []
                seen: Set[Tuple[str, str, str, str]] = set()
                with tar.extractfile(member) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = orjson.loads(line)
                        except Exception:
                            continue
                        for edge in self._parse_event(obj) or []:
                            if edge.actor_id not in self._node_type_map or edge.object_id not in self._node_type_map:
                                continue
                            key = (edge.actor_id, edge.object_id, edge.action, edge.timestamp)
                            if key in seen:
                                continue
                            seen.add(key)
                            edge.actor_type = self._node_type_map[edge.actor_id]
                            edge.object_type = self._node_type_map[edge.object_id]
                            edges.append(edge)
                            if len(edges) >= chunk_size:
                                total += len(edges)
                                self._write_edges(edges, output_dir, member.name)
                                edges = []
                if edges:
                    total += len(edges)
                    self._write_edges(edges, output_dir, member.name)
        return total

    def _parse_event(self, obj: Dict[str, Any]) -> List[ProvenanceEdge]:
        datum = obj.get("datum", {})
        if not datum:
            return []
        evt = datum.get(EVENT_TYPE)
        if not isinstance(evt, dict):
            return []
        action = evt.get("type", "")
        ts = str(evt.get("timestampNanos", ""))
        subj = evt.get("subject", {})
        actor_id = subj.get(UUID_KEY, "") if isinstance(subj, dict) else ""
        if not actor_id:
            return []
        props = evt.get("properties", {}).get("map", {}) if isinstance(evt.get("properties"), dict) else {}
        exec_path = props.get("exec", "")

        edges = []
        pred = evt.get("predicateObject", "")
        if isinstance(pred, dict) and (oid := pred.get(UUID_KEY)):
            fp = evt.get("predicateObjectPath", {})
            edges.append(ProvenanceEdge(actor_id, "", oid, "", action, ts, exec_path,
                                        fp.get("string", "") if isinstance(fp, dict) else ""))
        pred2 = evt.get("predicateObject2")
        if isinstance(pred2, dict) and (oid2 := pred2.get(UUID_KEY)):
            fp2 = evt.get("predicateObject2Path", {})
            edges.append(ProvenanceEdge(actor_id, "", oid2, "", action, ts, exec_path,
                                        fp2.get("string", "") if isinstance(fp2, dict) else ""))
        return edges

    def _write_edges(self, edges: List[ProvenanceEdge], output_dir: Path, source_name: str):
        base = Path(source_name).stem
        out = output_dir / f"{base}.csv.gz"
        mode = "at" if out.exists() else "wt"
        header = not out.exists()
        with gzip.open(out, mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EDGE_FIELDS)
            if header:
                writer.writeheader()
            writer.writerows([e.to_dict() for e in edges])


def parse_dataset(raw_data_dir: Path, output_dir: Path, archives: List[str]) -> Dict[str, Tuple[int, int]]:
    parser = CDMParser()
    results = {}
    for archive_name in archives:
        tar_path = raw_data_dir / archive_name
        if not tar_path.exists():
            continue
        n_nodes, n_edges = parser.parse_archive(tar_path, output_dir)
        results[archive_name] = (n_nodes, n_edges)
    return results
