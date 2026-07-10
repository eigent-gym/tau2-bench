"""Retrieval-tool compositions for the synthetic banking domain."""

from typing import TYPE_CHECKING

from tau2.domains.banking_knowledge.retrieval_mixins import (
    GrepMixin,
    KBSearchBm25AllToolsMixin,
    KBSearchDenseAllToolsMixin,
    KBSearchMixin,
    ShellMixin,
)
from tau2.domains.banking_knowledge_synthetic.tools import SyntheticBankingTools

if TYPE_CHECKING:
    from tau2.domains.banking_knowledge_synthetic.data_model import SyntheticBankingDB
    from tau2.knowledge.pipeline import RetrievalPipeline
    from tau2.knowledge.sandbox_manager import SandboxManager


class SyntheticBankingToolsPlain(SyntheticBankingTools):
    pass


class SyntheticBankingToolsWithKBSearch(KBSearchMixin, SyntheticBankingTools):
    def __init__(
        self,
        db: "SyntheticBankingDB",
        kb_pipeline: "RetrievalPipeline",
    ) -> None:
        super().__init__(db)
        self._kb_pipeline = kb_pipeline


class SyntheticBankingToolsWithGrep(GrepMixin, SyntheticBankingTools):
    def __init__(
        self,
        db: "SyntheticBankingDB",
        grep_pipeline: "RetrievalPipeline",
    ) -> None:
        super().__init__(db)
        self._grep_pipeline = grep_pipeline


class SyntheticBankingToolsWithKBSearchAndGrep(
    KBSearchMixin,
    GrepMixin,
    SyntheticBankingTools,
):
    def __init__(
        self,
        db: "SyntheticBankingDB",
        kb_pipeline: "RetrievalPipeline",
        grep_pipeline: "RetrievalPipeline",
    ) -> None:
        super().__init__(db)
        self._kb_pipeline = kb_pipeline
        self._grep_pipeline = grep_pipeline


class SyntheticBankingToolsWithShell(ShellMixin, SyntheticBankingTools):
    def __init__(
        self,
        db: "SyntheticBankingDB",
        sandbox: "SandboxManager",
    ) -> None:
        super().__init__(db)
        self._sandbox = sandbox


class SyntheticBankingToolsAllTools(
    KBSearchBm25AllToolsMixin,
    KBSearchDenseAllToolsMixin,
    ShellMixin,
    SyntheticBankingTools,
):
    def __init__(
        self,
        db: "SyntheticBankingDB",
        kb_bm25_pipeline: "RetrievalPipeline",
        kb_dense_pipeline: "RetrievalPipeline",
        sandbox: "SandboxManager",
    ) -> None:
        super().__init__(db)
        self._kb_bm25_pipeline = kb_bm25_pipeline
        self._kb_dense_pipeline = kb_dense_pipeline
        self._sandbox = sandbox
