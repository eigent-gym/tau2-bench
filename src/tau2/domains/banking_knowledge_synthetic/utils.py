"""Paths and deterministic identifiers for the synthetic banking domain."""

import hashlib

from tau2.utils.utils import DATA_DIR

SYNTHETIC_BANKING_DATA_DIR = (
    DATA_DIR / "tau2" / "domains" / "banking_knowledge_synthetic"
)
SYNTHETIC_BANKING_DOCUMENTS_DIR = SYNTHETIC_BANKING_DATA_DIR / "documents"
SYNTHETIC_BANKING_TASKS_DIR = SYNTHETIC_BANKING_DATA_DIR / "tasks"
SYNTHETIC_BANKING_DB_PATH = SYNTHETIC_BANKING_DATA_DIR / "db.json"
SYNTHETIC_BANKING_POLICY_PATH = SYNTHETIC_BANKING_DATA_DIR / "policy.md"


def deterministic_id(prefix: str, *parts: object, length: int = 16) -> str:
    """Return a stable identifier derived only from business inputs."""
    material = ":".join([prefix, *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"
