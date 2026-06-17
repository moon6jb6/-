"""模型注册表 — 内存存储，管理模型注册和版本信息。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """内存模型注册表。

    数据结构:
        _models: {model_id: {name, type, framework, created_at}}
        _versions: {model_id: [{version, file_hash, created_at}, ...]}
    """

    def __init__(self) -> None:
        self._models: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}

    def register(self, name: str, model_type: str, framework: str,
                 version: str, file_hash: str) -> Dict[str, Any]:
        """注册新模型，返回注册信息。"""
        model_id = "mdl_{}".format(uuid.uuid4().hex[:12])
        now = datetime.now(timezone.utc).isoformat()

        self._models[model_id] = {
            "model_id": model_id,
            "name": name,
            "type": model_type,
            "framework": framework,
            "created_at": now,
        }
        self._versions[model_id] = [
            {"version": version, "file_hash": file_hash, "created_at": now}
        ]

        return {
            "model_id": model_id,
            "name": name,
            "framework": framework,
            "version": version,
            "created_at": now,
        }

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型注册信息。"""
        return self._models.get(model_id)

    def get_versions(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型版本历史，不存在返回 None。"""
        if model_id not in self._models:
            return None
        return {
            "model_id": model_id,
            "name": self._models[model_id]["name"],
            "versions": self._versions.get(model_id, []),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有已注册模型。"""
        return list(self._models.values())


# 全局单例
model_registry = ModelRegistry()
