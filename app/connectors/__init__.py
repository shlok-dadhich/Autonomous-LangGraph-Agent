"""Connector registry — resolves SourceConnector by name."""

from __future__ import annotations

from typing import Dict, Type

from app.connectors.base import SourceConnector

_REGISTRY: dict[str, type[SourceConnector]] = {}


def register(name: str):
    def deco(cls: type[SourceConnector]):
        _REGISTRY[name] = cls
        return cls

    return deco


def get_connector(name: str) -> type[SourceConnector]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown connector: {name}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_connectors() -> list[str]:
    return sorted(_REGISTRY.keys())
