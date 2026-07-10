"""Database and knowledge-base models for synthetic banking tasks."""

from typing import Any

from pydantic import Field

from tau2.domains.banking_knowledge.data_model import (
    DatabaseTable,
    Document,
    KnowledgeBase,
)
from tau2.environment.db import DB


class SyntheticBankingDB(DB):
    """Shared transactional state for generated banking tasks."""

    users: DatabaseTable = Field(default_factory=DatabaseTable)
    accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    business_accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    debit_cards: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    bank_account_transactions: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_transaction_history: DatabaseTable = Field(
        default_factory=DatabaseTable
    )
    verification_history: DatabaseTable = Field(default_factory=DatabaseTable)
    human_transfer_requests: DatabaseTable = Field(default_factory=DatabaseTable)
    referrals: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_applications: DatabaseTable = Field(default_factory=DatabaseTable)
    transfers: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_transaction_disputes: DatabaseTable = Field(
        default_factory=DatabaseTable
    )
    debit_card_disputes: DatabaseTable = Field(default_factory=DatabaseTable)
    debit_dispute_intake_outcomes: DatabaseTable = Field(default_factory=DatabaseTable)
    written_dispute_statements: DatabaseTable = Field(default_factory=DatabaseTable)
    user_discoverable_tools: DatabaseTable = Field(default_factory=DatabaseTable)
    user_discoverable_tool_calls: DatabaseTable = Field(default_factory=DatabaseTable)
    agent_discoverable_tools: DatabaseTable = Field(default_factory=DatabaseTable)
    task_config: DatabaseTable = Field(default_factory=DatabaseTable)

    def get_statistics(self) -> dict[str, Any]:
        return {
            name: len(getattr(self, name).data) for name in self.__class__.model_fields
        }


__all__ = ["DatabaseTable", "Document", "KnowledgeBase", "SyntheticBankingDB"]
