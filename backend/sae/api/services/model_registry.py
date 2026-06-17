"""模型注册表 — JSON 文件持久化，管理模型注册和版本信息。"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_FILE = os.path.join(DATA_DIR, "models.json")


class ModelRegistry:
    """JSON 持久化模型注册表。"""

    def __init__(self) -> None:
        self._models: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(MODELS_FILE):
            return
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._models = data.get("models", {})
            self._versions = data.get("versions", {})
        except (json.JSONDecodeError, IOError):
            pass

    def _save(self) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {"models": self._models, "versions": self._versions}
        with open(MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def register(self, name, model_type, framework, version, file_hash,
                 sha256_verified=False):
        model_id = "mdl_{}".format(uuid.uuid4().hex[:12])
        now = datetime.now(timezone.utc).isoformat()
        self._models[model_id] = {
            "model_id": model_id,
            "name": name,
            "type": model_type,
            "framework": framework,
            "created_at": now,
        }
        self._versions[model_id] = [{
            "version": version,
            "file_hash": file_hash,
            "sha256_verified": sha256_verified,
            "created_at": now,
        }]
        self._save()
        return {
            "model_id": model_id,
            "name": name,
            "framework": framework,
            "version": version,
            "sha256_verified": sha256_verified,
            "created_at": now,
        }

    def get(self, model_id):
        return self._models.get(model_id)

    def get_versions(self, model_id):
        if model_id not in self._models:
            return None
        return {
            "model_id": model_id,
            "name": self._models[model_id]["name"],
            "versions": self._versions.get(model_id, []),
        }

    def list_models(self):
        return list(self._models.values())

    def delete(self, model_id):
        if model_id not in self._models:
            return False
        del self._models[model_id]
        self._versions.pop(model_id, None)
        self._save()
        return True


model_registry = ModelRegistry()
