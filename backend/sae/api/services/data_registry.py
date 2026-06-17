"""数据集注册表 — JSON 文件持久化。"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASETS_FILE = os.path.join(DATA_DIR, "datasets.json")


class DataRegistry:
    """内存+JSON持久化的数据集注册表。"""

    def __init__(self) -> None:
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(DATASETS_FILE):
            return
        try:
            with open(DATASETS_FILE, "r", encoding="utf-8") as f:
                self._datasets = json.load(f).get("datasets", {})
        except (json.JSONDecodeError, IOError):
            pass

    def _save(self) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(DATASETS_FILE, "w", encoding="utf-8") as f:
            json.dump({"datasets": self._datasets}, f, ensure_ascii=False, indent=2)

    def register(self, name, source, columns, row_count, lineage=None):
        dataset_id = "ds_{}".format(uuid.uuid4().hex[:12])
        now = datetime.now(timezone.utc).isoformat()
        self._datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "name": name,
            "source": source,
            "columns": columns,
            "row_count": row_count,
            "created_at": now,
            "lineage": lineage or {"parents": [], "transformations": [], "children": []},
        }
        self._save()
        return self._datasets[dataset_id]

    def get(self, dataset_id):
        return self._datasets.get(dataset_id)

    def get_lineage(self, dataset_id):
        ds = self._datasets.get(dataset_id)
        if ds is None:
            return None
        return ds.get("lineage", {"parents": [], "transformations": [], "children": []})

    def set_lineage(self, dataset_id, parents, transformations, children=None):
        if dataset_id not in self._datasets:
            return False
        self._datasets[dataset_id]["lineage"] = {
            "parents": parents,
            "transformations": transformations,
            "children": children or [],
        }
        self._save()
        return True

    def list_datasets(self):
        return list(self._datasets.values())

    def delete(self, dataset_id):
        if dataset_id not in self._datasets:
            return False
        del self._datasets[dataset_id]
        self._save()
        return True


data_registry = DataRegistry()
