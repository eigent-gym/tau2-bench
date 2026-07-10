"""Build configurable retrieval tools over the shared synthetic KB."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tau2.domains.banking_knowledge.retrieval import (
    DEFAULT_RETRIEVAL_VARIANT,
    RetrievalVariant,
    _create_grep_pipeline,
    _create_kb_pipeline,
    _create_sandbox,
    resolve_variant,
)
from tau2.domains.banking_knowledge_synthetic.data_model import (
    KnowledgeBase,
    SyntheticBankingDB,
)
from tau2.domains.banking_knowledge_synthetic.retrieval_toolkits import (
    SyntheticBankingToolsAllTools,
    SyntheticBankingToolsPlain,
    SyntheticBankingToolsWithGrep,
    SyntheticBankingToolsWithKBSearch,
    SyntheticBankingToolsWithKBSearchAndGrep,
    SyntheticBankingToolsWithShell,
)
from tau2.domains.banking_knowledge_synthetic.tools import SyntheticBankingTools
from tau2.domains.banking_knowledge_synthetic.utils import SYNTHETIC_BANKING_POLICY_PATH

if TYPE_CHECKING:
    from tau2.data_model.tasks import Task


def build_tools(
    variant: RetrievalVariant,
    db: SyntheticBankingDB,
    knowledge_base: KnowledgeBase,
) -> SyntheticBankingTools:
    """Compose synthetic banking tools with the selected retrieval capability."""
    if (
        variant.kb_search_bm25 is not None
        and variant.kb_search_dense is not None
        and variant.shell is not None
    ):
        return SyntheticBankingToolsAllTools(
            db,
            _create_kb_pipeline(variant.kb_search_bm25, knowledge_base),
            _create_kb_pipeline(variant.kb_search_dense, knowledge_base),
            _create_sandbox(knowledge_base, variant.shell),
        )

    has_kb = variant.kb_search is not None
    has_grep = variant.grep is not None
    if variant.shell is not None:
        return SyntheticBankingToolsWithShell(
            db,
            _create_sandbox(knowledge_base, variant.shell),
        )
    if has_kb and has_grep:
        return SyntheticBankingToolsWithKBSearchAndGrep(
            db,
            _create_kb_pipeline(variant.kb_search, knowledge_base),
            _create_grep_pipeline(variant.grep, knowledge_base),
        )
    if has_kb:
        return SyntheticBankingToolsWithKBSearch(
            db,
            _create_kb_pipeline(variant.kb_search, knowledge_base),
        )
    if has_grep:
        return SyntheticBankingToolsWithGrep(
            db,
            _create_grep_pipeline(variant.grep, knowledge_base),
        )
    return SyntheticBankingToolsPlain(db)


def _format_documents(documents: list[Any]) -> str:
    return "\n\n---\n\n".join(
        f"## {document.title}\n\n{document.content}" for document in documents
    )


def build_policy(
    variant: RetrievalVariant,
    knowledge_base: KnowledgeBase,
    task: "Task | None" = None,
) -> str:
    """Build a neutral domain policy plus retrieval-specific instructions."""
    policy = SYNTHETIC_BANKING_POLICY_PATH.read_text(encoding="utf-8").strip()
    retrieval_lines: list[str] = []
    if variant.kb_search_bm25 is not None:
        retrieval_lines.append("Use KB_search_bm25 for sparse retrieval.")
    if variant.kb_search_dense is not None:
        retrieval_lines.append("Use KB_search_dense for semantic retrieval.")
    if variant.kb_search is not None:
        retrieval_lines.append("Use KB_search to retrieve relevant documents.")
    if variant.grep is not None:
        retrieval_lines.append("Use grep for exact terms and tool names.")
    if variant.shell is not None:
        retrieval_lines.append("Use the read-only shell to inspect the KB corpus.")
    if retrieval_lines:
        policy += "\n\n## Knowledge Retrieval\n\n" + "\n".join(
            f"- {line}" for line in retrieval_lines
        )

    if variant.name == "full_kb":
        policy += "\n\n## Knowledge Base\n\n" + _format_documents(
            knowledge_base.get_all_documents()
        )
    elif variant.name == "golden_retrieval" and task is not None:
        required = [
            knowledge_base.get_document(document_id)
            for document_id in task.required_documents or []
        ]
        policy += "\n\n## Required Documents\n\n" + _format_documents(
            [document for document in required if document is not None]
        )
    return policy


def get_info_policy_override(
    variant_name: str | None,
    knowledge_base: KnowledgeBase,
    **kwargs: Any,
) -> str:
    variant = resolve_variant(variant_name or DEFAULT_RETRIEVAL_VARIANT, **kwargs)
    if variant.name == "golden_retrieval":
        return "(Policy is task-specific; inspect each simulation.)"
    return build_policy(variant, knowledge_base)
