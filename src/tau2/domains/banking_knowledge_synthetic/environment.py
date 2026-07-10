"""Environment factory for the synthetic banking knowledge domain."""

import json
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.retrieval import resolve_variant
from tau2.domains.banking_knowledge_synthetic.data_model import (
    KnowledgeBase,
    SyntheticBankingDB,
)
from tau2.domains.banking_knowledge_synthetic.retrieval import (
    build_policy,
    build_tools,
)
from tau2.domains.banking_knowledge_synthetic.tools import SyntheticBankingUserTools
from tau2.domains.banking_knowledge_synthetic.utils import (
    SYNTHETIC_BANKING_DB_PATH,
    SYNTHETIC_BANKING_DOCUMENTS_DIR,
    SYNTHETIC_BANKING_TASKS_DIR,
)
from tau2.environment.environment import Environment

DEFAULT_SYNTHETIC_RETRIEVAL_VARIANT = "bm25"


def get_db() -> SyntheticBankingDB:
    return SyntheticBankingDB.load(str(SYNTHETIC_BANKING_DB_PATH))


def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load(str(SYNTHETIC_BANKING_DOCUMENTS_DIR))


def get_environment(
    db: Optional[SyntheticBankingDB] = None,
    retrieval_variant: Optional[str] = None,
    retrieval_kwargs: Optional[dict] = None,
    task: Optional[Task] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("banking_knowledge_synthetic does not support solo mode")
    database = db or get_db()
    knowledge_base = get_knowledge_base()
    variant = resolve_variant(
        retrieval_variant or DEFAULT_SYNTHETIC_RETRIEVAL_VARIANT,
        **(retrieval_kwargs or {}),
    )
    return Environment(
        domain_name="banking_knowledge_synthetic",
        policy=build_policy(variant, knowledge_base, task),
        tools=build_tools(variant, database, knowledge_base),
        user_tools=SyntheticBankingUserTools(database),
    )


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    tasks: list[Task] = []
    for task_file in sorted(Path(SYNTHETIC_BANKING_TASKS_DIR).glob("task_*.json")):
        task = Task.model_validate(json.loads(task_file.read_text(encoding="utf-8")))
        tasks.append(task)
    if task_split_name in (None, "base"):
        return tasks
    return []


def get_tasks_split() -> dict[str, list[str]]:
    return {"base": [task.id for task in get_tasks()]}
