"""Agent and user tools for the synthetic banking knowledge domain."""

from __future__ import annotations

import inspect
import json
from typing import Any, Literal, get_type_hints

from pydantic import TypeAdapter, ValidationError

from tau2.domains.banking_knowledge.tools import (
    format_discoverable_tool_for_agent,
    parse_discoverable_tool_docstring,
)
from tau2.domains.banking_knowledge_synthetic.data_model import SyntheticBankingDB
from tau2.domains.banking_knowledge_synthetic.utils import deterministic_id
from tau2.environment.toolkit import (
    DISCOVERABLE_ATTR,
    ToolKitBase,
    ToolType,
    is_discoverable_tool,
    is_tool,
)


def _rows(db: SyntheticBankingDB, table: str) -> dict[str, dict[str, Any]]:
    return getattr(db, table).data


def _query(db: SyntheticBankingDB, table: str, **constraints: Any) -> str:
    matches = [
        record
        for record in _rows(db, table).values()
        if all(record.get(key) == value for key, value in constraints.items())
    ]
    return json.dumps(matches, indent=2, sort_keys=True)


def _insert(
    db: SyntheticBankingDB,
    table: str,
    record_id: str,
    record: dict[str, Any],
) -> bool:
    records = _rows(db, table)
    if record_id in records:
        return False
    records[record_id] = record
    return True


def _task_clock(db: SyntheticBankingDB) -> dict[str, str]:
    return db.task_config.data.get(
        "scenario_clock",
        {
            "current_date": "07/10/2026",
            "current_time": "2026-07-10 00:00:00 EST",
        },
    )


def _validate_tool_arguments(
    method: Any,
    arguments: dict[str, Any],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate wrapper arguments against the implementation annotations."""
    signature = inspect.signature(method)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    }
    unknown = set(arguments) - set(parameters)
    if unknown:
        raise ValueError(f"Unexpected parameter(s): {', '.join(sorted(unknown))}")

    if not allow_partial:
        try:
            signature.bind(**arguments)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    type_hints = get_type_hints(method)
    validated: dict[str, Any] = {}
    for name, value in arguments.items():
        annotation = type_hints.get(name)
        if annotation is None:
            validated[name] = value
            continue
        try:
            validated[name] = TypeAdapter(annotation).validate_python(value)
        except ValidationError as exc:
            message = exc.errors()[0]["msg"]
            raise ValueError(f"Invalid value for '{name}': {message}") from exc
    return validated


class SyntheticBankingTools(ToolKitBase):
    """Tools available to the tested customer-service agent."""

    db: SyntheticBankingDB

    def __init__(self, db: SyntheticBankingDB) -> None:
        super().__init__(db)
        self._unlocked_agent_tools: set[str] = set()

    @is_tool(ToolType.READ)
    def get_current_time(self) -> str:
        """Return the task's fixed scenario time."""
        return _task_clock(self.db)["current_time"]

    @is_tool(ToolType.READ)
    def get_user_information_by_id(self, user_id: str) -> str:
        """Look up a customer by user ID.

        Args:
            user_id: Customer identifier.
        """
        return _query(self.db, "users", user_id=user_id)

    @is_tool(ToolType.READ)
    def get_user_information_by_name(self, customer_name: str) -> str:
        """Look up a customer by exact full name.

        Args:
            customer_name: Customer's full name.
        """
        return _query(self.db, "users", name=customer_name)

    @is_tool(ToolType.READ)
    def get_user_information_by_email(self, email: str) -> str:
        """Look up a customer by email address.

        Args:
            email: Customer's email address.
        """
        return _query(self.db, "users", email=email)

    @is_tool(ToolType.READ)
    def get_accounts_by_user(self, user_id: str) -> str:
        """Return deposit accounts belonging to a customer.

        Args:
            user_id: Customer identifier.
        """
        return _query(self.db, "accounts", user_id=user_id)

    @is_tool(ToolType.READ)
    def get_credit_card_accounts_by_user(self, user_id: str) -> str:
        """Return credit-card accounts without exposing card digits.

        Args:
            user_id: Customer identifier.
        """
        records = json.loads(_query(self.db, "credit_card_accounts", user_id=user_id))
        for record in records:
            record.pop("last_4_digits", None)
            record.pop("card_last_4_digits", None)
        return json.dumps(records, indent=2, sort_keys=True)

    @is_tool(ToolType.READ)
    def get_debit_cards_by_account(self, account_id: str) -> str:
        """Return debit cards linked to an account without exposing card digits.

        Args:
            account_id: Deposit account identifier.
        """
        records = json.loads(_query(self.db, "debit_cards", account_id=account_id))
        for record in records:
            record.pop("last_4_digits", None)
            record.pop("card_last_4_digits", None)
        return json.dumps(records, indent=2, sort_keys=True)

    @is_tool(ToolType.READ)
    def get_bank_account_transactions_by_user(self, user_id: str) -> str:
        """Return deposit-account transactions belonging to a customer.

        Args:
            user_id: Customer identifier.
        """
        return _query(self.db, "bank_account_transactions", user_id=user_id)

    @is_tool(ToolType.READ)
    def get_credit_card_transactions_by_user(self, user_id: str) -> str:
        """Return credit-card transactions belonging to a customer.

        Args:
            user_id: Customer identifier.
        """
        return _query(self.db, "credit_card_transaction_history", user_id=user_id)

    @is_tool(ToolType.WRITE)
    def log_verification(
        self,
        name: str,
        user_id: str,
        address: str,
        email: str,
        phone_number: str,
        date_of_birth: str,
        time_verified: str,
    ) -> str:
        """Record a completed identity verification.

        Args:
            name: Customer's full name.
            user_id: Customer identifier.
            address: Verified address.
            email: Verified email.
            phone_number: Verified phone number.
            date_of_birth: Verified date of birth.
            time_verified: Scenario timestamp returned by get_current_time.
        """
        record_id = deterministic_id("verification", user_id, time_verified)
        record = {
            "name": name,
            "user_id": user_id,
            "address": address,
            "email": email,
            "phone_number": phone_number,
            "date_of_birth": date_of_birth,
            "time_verified": time_verified,
        }
        if not _insert(self.db, "verification_history", record_id, record):
            return f"Verification record {record_id} already exists."
        return f"Verification logged for {name} at {time_verified}."

    @is_tool(ToolType.WRITE)
    def transfer_to_human_agents(
        self,
        reason: str,
        summary: str = "",
        user_id: str | None = None,
    ) -> str:
        """Record a transfer to a human support agent.

        Args:
            reason: Policy-defined transfer reason.
            summary: Short summary for the receiving agent.
            user_id: Customer identifier when known.
        """
        record_id = deterministic_id(
            "human_transfer", user_id or "unknown", reason, summary
        )
        record = {
            "transfer_id": record_id,
            "reason": reason,
            "summary": summary,
            "user_id": user_id,
        }
        _insert(self.db, "human_transfer_requests", record_id, record)
        return f"Human transfer requested with reason {reason}."

    @is_tool(ToolType.WRITE)
    def record_debit_dispute_intake_outcome(
        self,
        transaction_id: str,
        account_id: str,
        user_id: str,
        outcome: Literal[
            "explain_pending_transaction_not_disputable_yet",
            "deny_not_timely_or_escalate",
            "explain_open_dispute_limit",
            "ask_user_to_contact_merchant_first",
        ],
        required_next_step: Literal[
            "ask_user_for_merchant_contact",
            "do_not_call_hidden_agent_tool",
        ],
        transaction_status: Literal["pending", "posted"] | None = None,
        contacted_merchant: bool | None = None,
        current_open_disputes: int | None = None,
        maximum_open_disputes: int | None = None,
        notice_is_within_statement_window: bool | None = None,
    ) -> str:
        """Record a policy-grounded debit-dispute result that does not file a dispute.

        Args:
            transaction_id: Transaction reviewed during intake.
            account_id: Deposit account linked to the transaction.
            user_id: Customer identifier.
            outcome: Policy-defined non-filing outcome.
            required_next_step: Follow-up required by the policy outcome.
            transaction_status: Pending or posted status when relevant.
            contacted_merchant: Whether merchant contact has occurred when relevant.
            current_open_disputes: Current open-dispute count when relevant.
            maximum_open_disputes: Account-tier dispute limit when relevant.
            notice_is_within_statement_window: Timeliness result when relevant.
        """
        transaction = self.db.bank_account_transactions.data.get(transaction_id)
        if transaction is None:
            return f"Error: Transaction '{transaction_id}' not found."
        if transaction.get("account_id") != account_id:
            return "Error: Transaction is not linked to the supplied account."
        if transaction.get("user_id") != user_id:
            return "Error: Transaction is not linked to the supplied customer."

        expected_next_step = (
            "ask_user_for_merchant_contact"
            if outcome == "ask_user_to_contact_merchant_first"
            else "do_not_call_hidden_agent_tool"
        )
        if required_next_step != expected_next_step:
            return f"Error: Outcome requires required_next_step='{expected_next_step}'."
        if (
            outcome == "explain_pending_transaction_not_disputable_yet"
            and transaction_status != "pending"
        ):
            return "Error: Pending-transaction outcome requires transaction_status='pending'."
        if (
            outcome == "ask_user_to_contact_merchant_first"
            and contacted_merchant is not False
        ):
            return "Error: Merchant-contact outcome requires contacted_merchant=false."
        if outcome == "explain_open_dispute_limit" and (
            current_open_disputes is None
            or maximum_open_disputes is None
            or current_open_disputes < maximum_open_disputes
        ):
            return "Error: Open-dispute-limit outcome requires a reached account limit."
        if (
            outcome == "deny_not_timely_or_escalate"
            and notice_is_within_statement_window is not False
        ):
            return "Error: Late-notice outcome requires an expired statement window."

        outcome_id = deterministic_id("debit_intake_outcome", transaction_id, user_id)
        evidence = {
            "transaction_status": None,
            "contacted_merchant": None,
            "current_open_disputes": None,
            "maximum_open_disputes": None,
            "notice_is_within_statement_window": None,
        }
        if outcome == "explain_pending_transaction_not_disputable_yet":
            evidence["transaction_status"] = transaction_status
        elif outcome == "ask_user_to_contact_merchant_first":
            evidence["contacted_merchant"] = contacted_merchant
        elif outcome == "explain_open_dispute_limit":
            evidence["current_open_disputes"] = current_open_disputes
            evidence["maximum_open_disputes"] = maximum_open_disputes
        elif outcome == "deny_not_timely_or_escalate":
            evidence["notice_is_within_statement_window"] = (
                notice_is_within_statement_window
            )
        record = {
            "outcome_id": outcome_id,
            "transaction_id": transaction_id,
            "account_id": account_id,
            "user_id": user_id,
            "outcome": outcome,
            "required_next_step": required_next_step,
            **evidence,
        }
        if not _insert(
            self.db,
            "debit_dispute_intake_outcomes",
            outcome_id,
            record,
        ):
            return f"Debit-dispute intake outcome {outcome_id} already exists."
        return f"Debit-dispute intake outcome recorded. Outcome ID: {outcome_id}."

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def give_discoverable_user_tool(
        self,
        discoverable_tool_name: str,
        arguments: str = "{}",
    ) -> str:
        """Give a documented discoverable tool to the simulated customer.

        Args:
            discoverable_tool_name: Exact user tool name found in the KB.
            arguments: Optional JSON object containing prefilled arguments.
        """
        method = getattr(SyntheticBankingUserTools, discoverable_tool_name, None)
        if method is None or not getattr(method, DISCOVERABLE_ATTR, False):
            return f"Error: Unknown discoverable user tool '{discoverable_tool_name}'."
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON arguments: {exc}"
        if not isinstance(parsed_arguments, dict):
            return "Error: Tool arguments must be a JSON object."
        try:
            parsed_arguments = _validate_tool_arguments(
                method,
                parsed_arguments,
                allow_partial=True,
            )
        except ValueError as exc:
            return f"Error: Invalid prefilled arguments: {exc}"
        record_id = deterministic_id("user_tool", discoverable_tool_name)
        self.db.user_discoverable_tools.data[record_id] = {
            "tool_name": discoverable_tool_name,
            "status": "GIVEN",
        }
        tool_info = parse_discoverable_tool_docstring(method)
        prefilled = json.dumps(parsed_arguments, indent=2, sort_keys=True)
        return (
            f"Tool given to user: {discoverable_tool_name}\n"
            f"{format_discoverable_tool_for_agent(tool_info)}\n"
            f"Prefilled arguments: {prefilled}\n\n"
            "Tell the user to call `call_discoverable_user_tool` with "
            f"discoverable_tool_name='{discoverable_tool_name}' and a JSON object "
            "containing every required parameter shown above."
        )

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def unlock_discoverable_agent_tool(self, agent_tool_name: str) -> str:
        """Unlock an internal tool whose exact name was found in the KB.

        Args:
            agent_tool_name: Exact hidden agent tool name.
        """
        if not self.has_discoverable_tool(agent_tool_name):
            return f"Error: Unknown agent tool '{agent_tool_name}'."
        self._unlocked_agent_tools.add(agent_tool_name)
        method = self.get_discoverable_tools()[agent_tool_name]
        info = parse_discoverable_tool_docstring(method)
        return (
            f"Tool unlocked: {agent_tool_name}\n"
            f"{format_discoverable_tool_for_agent(info)}"
        )

    @is_tool(ToolType.WRITE)
    def call_discoverable_agent_tool(
        self,
        agent_tool_name: str,
        arguments: str = "{}",
    ) -> str:
        """Call a previously unlocked internal agent tool.

        Args:
            agent_tool_name: Exact hidden agent tool name.
            arguments: JSON object containing the documented arguments.
        """
        if agent_tool_name not in self._unlocked_agent_tools:
            return f"Error: Tool '{agent_tool_name}' has not been unlocked."
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON arguments: {exc}"
        if not isinstance(parsed_arguments, dict):
            return "Error: Tool arguments must be a JSON object."
        method = self.get_discoverable_tools()[agent_tool_name]
        try:
            validated_arguments = _validate_tool_arguments(method, parsed_arguments)
            result = method(**validated_arguments)
        except ValueError as exc:
            return f"Error: Invalid arguments for '{agent_tool_name}': {exc}"
        record_id = deterministic_id("agent_tool", agent_tool_name)
        self.db.agent_discoverable_tools.data[record_id] = {
            "tool_name": agent_tool_name,
            "status": "CALLED",
        }
        return result

    @is_discoverable_tool(ToolType.WRITE)
    def file_credit_card_transaction_dispute_7383(
        self,
        transaction_id: str,
        card_action: Literal["keep_active", "cancel_and_reissue"],
        card_last_4_digits: str,
        full_name: str,
        user_id: str,
        phone: str,
        email: str,
        address: str,
        contacted_merchant: bool,
        purchase_date: str,
        issue_noticed_date: str,
        dispute_reason: str,
        resolution_requested: Literal["full_refund", "partial_refund"],
        eligible_for_provisional_credit: bool,
        partial_refund_amount: float | None = None,
    ) -> str:
        """File a credit-card transaction dispute.

        Args:
            transaction_id (string): Transaction being disputed.
            card_action (string): Keep active or cancel and reissue.
            card_last_4_digits (string): Customer-provided card digits.
            full_name (string): Customer's full name.
            user_id (string): Customer identifier.
            phone (string): Registered phone number.
            email (string): Registered email address.
            address (string): Registered address.
            contacted_merchant (boolean): Whether merchant contact occurred.
            purchase_date (string): Purchase date.
            issue_noticed_date (string): Date the issue was noticed.
            dispute_reason (string): Policy-defined dispute reason.
            resolution_requested (string): Full or partial refund.
            eligible_for_provisional_credit (boolean): Policy decision.
            partial_refund_amount (number, optional): Requested partial amount.
        """
        if resolution_requested == "partial_refund" and partial_refund_amount is None:
            return "Error: partial_refund_amount is required for a partial refund."
        dispute_id = deterministic_id("cc_dispute", transaction_id, user_id)
        record = dict(locals())
        record.pop("self")
        record.update({"dispute_id": dispute_id, "status": "OPEN"})
        if not _insert(
            self.db,
            "credit_card_transaction_disputes",
            dispute_id,
            record,
        ):
            return f"Credit-card dispute {dispute_id} already exists."
        return f"Credit-card transaction dispute filed. Dispute ID: {dispute_id}."

    @is_discoverable_tool(ToolType.WRITE)
    def file_debit_card_transaction_dispute_5724(
        self,
        transaction_id: str,
        account_id: str,
        card_id: str,
        user_id: str,
        dispute_category: str,
        transaction_date: str,
        discovery_date: str,
        disputed_amount: float,
        transaction_type: str,
        card_in_possession: bool,
        pin_compromised: Literal["yes_shared", "yes_observed", "no", "unknown"],
        contacted_merchant: bool,
        police_report_filed: bool,
        written_statement_provided: bool,
        provisional_credit_eligible: bool,
        customer_max_liability_amount: float,
        card_action: Literal[
            "keep_active", "freeze_pending_investigation", "close_and_reissue"
        ],
    ) -> str:
        """File a debit-card transaction dispute.

        Args:
            transaction_id (string): Transaction being disputed.
            account_id (string): Linked checking account.
            card_id (string): Debit-card identifier.
            user_id (string): Customer identifier.
            dispute_category (string): Regulation E dispute category.
            transaction_date (string): Transaction date.
            discovery_date (string): Date the issue was discovered.
            disputed_amount (number): Amount disputed.
            transaction_type (string): Transaction channel.
            card_in_possession (boolean): Whether customer has the card.
            pin_compromised (string): PIN compromise status.
            contacted_merchant (boolean): Whether merchant contact occurred.
            police_report_filed (boolean): Whether a report was filed.
            written_statement_provided (boolean): Written statement status.
            provisional_credit_eligible (boolean): Policy decision.
            customer_max_liability_amount (number): Computed liability cap.
            card_action (string): Required card action.
        """
        dispute_id = deterministic_id("debit_dispute", transaction_id, user_id)
        record = dict(locals())
        record.pop("self")
        record.update({"dispute_id": dispute_id, "status": "OPEN"})
        if not _insert(self.db, "debit_card_disputes", dispute_id, record):
            return f"Debit-card dispute {dispute_id} already exists."
        return f"Debit-card transaction dispute filed. Dispute ID: {dispute_id}."

    @is_discoverable_tool(ToolType.WRITE)
    def open_bank_account_5464(
        self,
        user_id: str,
        account_type: str,
        account_class: str,
    ) -> str:
        """Open a synthetic personal deposit account.

        Args:
            user_id (string): Customer identifier.
            account_type (string): Checking or savings.
            account_class (string): Selected product class.
        """
        account_id = deterministic_id("account", user_id, account_type, account_class)
        record = {
            "account_id": account_id,
            "user_id": user_id,
            "account_type": account_type,
            "account_class": account_class,
            "status": "OPEN",
            "date_opened": _task_clock(self.db)["current_date"],
        }
        if not _insert(self.db, "accounts", account_id, record):
            return f"Account {account_id} already exists."
        return f"Deposit account opened. Account ID: {account_id}."

    @is_discoverable_tool(ToolType.WRITE)
    def close_bank_account_1097(
        self,
        account_id: str,
        reason: str | None = None,
        waive_early_closure_fee: bool | None = None,
    ) -> str:
        """Close an eligible synthetic deposit account.

        Args:
            account_id (string): Account to close.
            reason (string, optional): Customer's closure reason.
            waive_early_closure_fee (boolean, optional): Approved waiver decision.
        """
        account = self.db.accounts.data.get(account_id)
        if account is None:
            return f"Error: Account '{account_id}' not found."
        account.update(
            {
                "status": "CLOSED",
                "date_closed": _task_clock(self.db)["current_date"],
                "closure_reason": reason,
                "early_closure_fee_waived": waive_early_closure_fee,
            }
        )
        return f"Deposit account {account_id} closed."

    @is_discoverable_tool(ToolType.WRITE)
    def open_business_deposit_account_8514(
        self,
        user_id: str,
        business_id: str,
        account_class: str,
        initial_deposit: float,
        authorized_signer_name: str,
    ) -> str:
        """Open an eligible business deposit account.

        Args:
            user_id (string): Customer identifier.
            business_id (string): Business identifier.
            account_class (string): Selected product class.
            initial_deposit (number): Opening deposit.
            authorized_signer_name (string): Authorized signer.
        """
        account_id = deterministic_id("business_account", business_id, account_class)
        record = {
            "account_id": account_id,
            "user_id": user_id,
            "business_id": business_id,
            "account_class": account_class,
            "initial_deposit": initial_deposit,
            "authorized_signer_name": authorized_signer_name,
            "status": "OPEN",
            "date_opened": _task_clock(self.db)["current_date"],
        }
        if not _insert(self.db, "business_accounts", account_id, record):
            return f"Business account {account_id} already exists."
        return f"Business deposit account opened. Account ID: {account_id}."

    @is_discoverable_tool(ToolType.WRITE)
    def submit_customer_transfer_2674(
        self,
        user_id: str,
        account_id: str,
        transfer_type: str,
        amount: float,
        destination_reference: str,
        effective_date: str,
    ) -> str:
        """Submit an eligible customer transfer.

        Args:
            user_id (string): Customer identifier.
            account_id (string): Funding account.
            transfer_type (string): Transfer rail.
            amount (number): Transfer amount.
            destination_reference (string): Destination identifier.
            effective_date (string): Requested effective date.
        """
        transfer_id = deterministic_id(
            "transfer",
            user_id,
            account_id,
            destination_reference,
            amount,
            effective_date,
        )
        record = {
            "transfer_id": transfer_id,
            "user_id": user_id,
            "account_id": account_id,
            "transfer_type": transfer_type,
            "amount": amount,
            "destination_reference": destination_reference,
            "effective_date": effective_date,
            "status": "SUBMITTED",
        }
        if not _insert(self.db, "transfers", transfer_id, record):
            return f"Transfer {transfer_id} already exists."
        return f"Transfer submitted. Transfer ID: {transfer_id}."


class SyntheticBankingUserTools(ToolKitBase):
    """Tools available only to the simulated customer."""

    db: SyntheticBankingDB

    def _was_given(self, tool_name: str) -> bool:
        return any(
            record.get("tool_name") == tool_name
            for record in self.db.user_discoverable_tools.data.values()
        )

    def _log_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        record_id = deterministic_id(
            "user_tool_call", tool_name, json.dumps(arguments, sort_keys=True)
        )
        self.db.user_discoverable_tool_calls.data[record_id] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "CALLED",
        }

    @is_tool(ToolType.WRITE)
    def call_discoverable_user_tool(
        self,
        discoverable_tool_name: str,
        arguments: str = "{}",
    ) -> str:
        """Call a user tool previously provided by the support agent.

        Args:
            discoverable_tool_name: Exact tool name supplied by the agent.
            arguments: JSON object containing the tool arguments.
        """
        if not self.has_discoverable_tool(discoverable_tool_name):
            return f"Error: Unknown user tool '{discoverable_tool_name}'."
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON arguments: {exc}"
        if not isinstance(parsed_arguments, dict):
            return "Error: Tool arguments must be a JSON object."
        method = self.get_discoverable_tools()[discoverable_tool_name]
        try:
            validated_arguments = _validate_tool_arguments(method, parsed_arguments)
        except ValueError as exc:
            return f"Error: Invalid arguments for '{discoverable_tool_name}': {exc}"
        return method(**validated_arguments)

    @is_discoverable_tool(ToolType.READ, mutates_state=True)
    def get_card_last_4_digits(self, credit_card_account_id: str) -> str:
        """Retrieve the last four digits of a credit card.

        Args:
            credit_card_account_id (string): Credit-card account identifier.
        """
        if not self._was_given("get_card_last_4_digits"):
            return "Error: The support agent has not provided this tool."
        account = self.db.credit_card_accounts.data.get(credit_card_account_id)
        if account is None:
            return f"Error: Credit-card account '{credit_card_account_id}' not found."
        arguments = {"credit_card_account_id": credit_card_account_id}
        self._log_call("get_card_last_4_digits", arguments)
        digits = account.get("last_4_digits", account.get("card_last_4_digits"))
        return json.dumps({"card_last_4_digits": digits})

    @is_discoverable_tool(ToolType.READ, mutates_state=True)
    def get_debit_card_last_4_digits(self, card_id: str) -> str:
        """Retrieve the last four digits of a debit card.

        Args:
            card_id (string): Debit-card identifier.
        """
        if not self._was_given("get_debit_card_last_4_digits"):
            return "Error: The support agent has not provided this tool."
        card = self.db.debit_cards.data.get(card_id)
        if card is None:
            return f"Error: Debit card '{card_id}' not found."
        arguments = {"card_id": card_id}
        self._log_call("get_debit_card_last_4_digits", arguments)
        digits = card.get("last_4_digits", card.get("card_last_4_digits"))
        return json.dumps({"card_last_4_digits": digits})

    @is_discoverable_tool(ToolType.WRITE)
    def provide_written_dispute_statement(
        self,
        account_id: str,
        confirm_written_statement: bool,
    ) -> str:
        """Confirm that the conversation serves as a written dispute statement.

        Args:
            account_id (string): Account associated with the dispute.
            confirm_written_statement (boolean): Customer confirmation.
        """
        if not self._was_given("provide_written_dispute_statement"):
            return "Error: The support agent has not provided this tool."
        arguments = {
            "account_id": account_id,
            "confirm_written_statement": confirm_written_statement,
        }
        self._log_call("provide_written_dispute_statement", arguments)
        record_id = deterministic_id("written_statement", account_id)
        self.db.written_dispute_statements.data[record_id] = {
            "statement_id": record_id,
            "account_id": account_id,
            "written_statement_provided": confirm_written_statement,
        }
        return json.dumps({"written_statement_provided": confirm_written_statement})

    @is_tool(ToolType.WRITE)
    def apply_for_credit_card(
        self,
        card_type: str,
        customer_name: str,
        annual_income: float,
        rho_bank_subscription: bool,
    ) -> str:
        """Submit the customer's selected credit-card application.

        Args:
            card_type: Selected card product.
            customer_name: Applicant's full legal name.
            annual_income: Applicant's annual income.
            rho_bank_subscription: Whether the applicant has the subscription.
        """
        application_id = deterministic_id(
            "application",
            card_type,
            customer_name,
            annual_income,
            rho_bank_subscription,
        )
        record = {
            "application_id": application_id,
            "card_type": card_type,
            "customer_name": customer_name,
            "annual_income": annual_income,
            "rho_bank_subscription": rho_bank_subscription,
            "status": "PENDING",
            "date": _task_clock(self.db)["current_date"],
        }
        if not _insert(self.db, "credit_card_applications", application_id, record):
            return f"Application {application_id} already exists."
        return f"Credit-card application submitted. Application ID: {application_id}."

    @is_tool(ToolType.WRITE)
    def submit_referral(self, user_id: str, account_type: str) -> str:
        """Submit a customer referral for a selected account program.

        Args:
            user_id: Referring customer's identifier.
            account_type: Selected referral account type.
        """
        referral_id = deterministic_id("referral", user_id, account_type)
        record = {
            "referral_id": referral_id,
            "referrer_id": user_id,
            "referred_account_type": account_type,
            "referral_status": "NO_PROGRESS",
            "date": _task_clock(self.db)["current_date"],
        }
        if not _insert(self.db, "referrals", referral_id, record):
            return f"Referral {referral_id} already exists."
        return f"Referral submitted. Referral ID: {referral_id}."
