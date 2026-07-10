import json

from tau2.data_model.tasks import InitializationData
from tau2.domains.banking_knowledge_synthetic.environment import (
    get_db,
    get_environment,
    get_knowledge_base,
)
from tau2.domains.banking_knowledge_synthetic.utils import (
    SYNTHETIC_BANKING_DATA_DIR,
)
from tau2.registry import registry


def test_shared_world_loads_and_is_registered() -> None:
    database = get_db()
    knowledge_base = get_knowledge_base()

    assert len(knowledge_base.documents) == 91
    assert len(database.users.data) == 32
    assert "banking_knowledge_synthetic" in registry.get_domains()
    assert "banking_knowledge_synthetic" in registry.get_task_sets()


def test_bm25_environment_exposes_retrieval_and_wrappers() -> None:
    environment = get_environment(retrieval_variant="bm25")
    tool_names = {tool.name for tool in environment.get_tools()}

    assert environment.domain_name == "banking_knowledge_synthetic"
    assert "KB_search" in tool_names
    assert "unlock_discoverable_agent_tool" in tool_names
    assert "call_discoverable_agent_tool" in tool_names
    assert "record_debit_dispute_intake_outcome" in tool_names
    assert "file_debit_card_transaction_dispute_5724" not in tool_names

    result = environment.make_tool_call(
        "KB_search",
        query="debit card provisional credit",
    )
    assert "Debit Card" in result


def test_hidden_agent_tool_call_mutates_shared_db() -> None:
    environment = get_environment(retrieval_variant="bm25")
    database = environment.tools.db
    transaction = next(iter(database.bank_account_transactions.data.values()))
    account = database.accounts.data[transaction["account_id"]]
    card = next(
        item
        for item in database.debit_cards.data.values()
        if item["account_id"] == account["account_id"]
    )

    unlocked = environment.make_tool_call(
        "unlock_discoverable_agent_tool",
        agent_tool_name="file_debit_card_transaction_dispute_5724",
    )
    assert "Tool unlocked" in unlocked

    arguments = {
        "transaction_id": transaction["transaction_id"],
        "account_id": account["account_id"],
        "card_id": card["card_id"],
        "user_id": transaction["user_id"],
        "dispute_category": "card_present_fraud",
        "transaction_date": transaction["transaction_date"],
        "discovery_date": "2026-07-07",
        "disputed_amount": transaction["transaction_amount"],
        "transaction_type": "pin_purchase",
        "card_in_possession": False,
        "pin_compromised": "yes_observed",
        "contacted_merchant": False,
        "police_report_filed": False,
        "written_statement_provided": True,
        "provisional_credit_eligible": True,
        "customer_max_liability_amount": 50.0,
        "card_action": "close_and_reissue",
    }
    result = environment.make_tool_call(
        "call_discoverable_agent_tool",
        agent_tool_name="file_debit_card_transaction_dispute_5724",
        arguments=json.dumps(arguments),
    )

    assert "dispute filed" in result
    assert any(
        record["transaction_id"] == transaction["transaction_id"]
        for record in database.debit_card_disputes.data.values()
    )


def test_user_discoverable_tool_uses_same_database() -> None:
    environment = get_environment(retrieval_variant="bm25")
    account_id = "synthetic_cc_test"
    environment.tools.db.credit_card_accounts.data[account_id] = {
        "account_id": account_id,
        "user_id": "syn_debit_user_001",
        "card_last_4_digits": "4821",
    }

    grant_result = environment.make_tool_call(
        "give_discoverable_user_tool",
        discoverable_tool_name="get_card_last_4_digits",
    )
    assert "credit_card_account_id: string (required)" in grant_result
    assert "call_discoverable_user_tool" in grant_result
    result = environment.make_tool_call(
        "call_discoverable_user_tool",
        requestor="user",
        discoverable_tool_name="get_card_last_4_digits",
        arguments=json.dumps({"credit_card_account_id": account_id}),
    )

    assert json.loads(result) == {"card_last_4_digits": "4821"}
    assert environment.tools.db is environment.user_tools.db
    assert environment.tools.db.user_discoverable_tool_calls.data


def test_hidden_tool_wrapper_validates_and_normalizes_arguments() -> None:
    environment = get_environment(retrieval_variant="bm25")
    database = environment.tools.db
    transaction = next(iter(database.bank_account_transactions.data.values()))
    account = database.accounts.data[transaction["account_id"]]
    card = next(
        item
        for item in database.debit_cards.data.values()
        if item["account_id"] == account["account_id"]
    )
    environment.make_tool_call(
        "unlock_discoverable_agent_tool",
        agent_tool_name="file_debit_card_transaction_dispute_5724",
    )
    arguments = {
        "transaction_id": transaction["transaction_id"],
        "account_id": account["account_id"],
        "card_id": card["card_id"],
        "user_id": transaction["user_id"],
        "dispute_category": "card_present_fraud",
        "transaction_date": transaction["transaction_date"],
        "discovery_date": "2026-07-07",
        "disputed_amount": transaction["transaction_amount"],
        "transaction_type": "pin_purchase",
        "card_in_possession": False,
        "pin_compromised": "observed",
        "contacted_merchant": False,
        "police_report_filed": False,
        "written_statement_provided": True,
        "provisional_credit_eligible": True,
        "customer_max_liability_amount": 50,
        "card_action": "close_and_reissue",
    }

    invalid_result = environment.make_tool_call(
        "call_discoverable_agent_tool",
        agent_tool_name="file_debit_card_transaction_dispute_5724",
        arguments=json.dumps(arguments),
    )
    assert "Invalid value for 'pin_compromised'" in invalid_result
    assert not database.debit_card_disputes.data

    arguments["pin_compromised"] = "yes_observed"
    valid_result = environment.make_tool_call(
        "call_discoverable_agent_tool",
        agent_tool_name="file_debit_card_transaction_dispute_5724",
        arguments=json.dumps(arguments),
    )
    assert "dispute filed" in valid_result
    record = next(iter(database.debit_card_disputes.data.values()))
    assert record["customer_max_liability_amount"] == 50.0
    assert isinstance(record["customer_max_liability_amount"], float)


def test_non_filing_debit_outcome_is_programmatically_recorded() -> None:
    environment = get_environment(retrieval_variant="bm25")
    database = environment.tools.db
    transaction = next(iter(database.bank_account_transactions.data.values()))
    account = database.accounts.data[transaction["account_id"]]
    arguments = {
        "transaction_id": transaction["transaction_id"],
        "account_id": account["account_id"],
        "user_id": transaction["user_id"],
        "outcome": "ask_user_to_contact_merchant_first",
        "required_next_step": "ask_user_for_merchant_contact",
        "contacted_merchant": False,
    }

    result = environment.make_tool_call(
        "record_debit_dispute_intake_outcome",
        **arguments,
    )
    assert "outcome recorded" in result
    record = next(iter(database.debit_dispute_intake_outcomes.data.values()))
    assert record["outcome"] == "ask_user_to_contact_merchant_first"
    assert record["contacted_merchant"] is False

    before = len(database.debit_dispute_intake_outcomes.data)
    invalid_result = environment.make_tool_call(
        "record_debit_dispute_intake_outcome",
        transaction_id=transaction["transaction_id"],
        account_id=account["account_id"],
        user_id=transaction["user_id"],
        outcome="explain_open_dispute_limit",
        required_next_step="do_not_call_hidden_agent_tool",
        current_open_disputes=1,
        maximum_open_disputes=2,
    )
    assert "requires a reached account limit" in invalid_result
    assert len(database.debit_dispute_intake_outcomes.data) == before


def test_task_overlay_is_isolated_and_shared_between_toolkits() -> None:
    first = get_environment(retrieval_variant="bm25")
    second = get_environment(retrieval_variant="bm25")
    overlay = InitializationData.model_validate(
        {
            "agent_data": {
                "task_config": {
                    "data": {
                        "scenario_clock": {
                            "current_date": "07/08/2026",
                            "current_time": "2026-07-08 14:05:00 EST",
                        }
                    }
                }
            }
        }
    )

    first.set_state(overlay, initialization_actions=None, message_history=[])

    assert first.make_tool_call("get_current_time") == "2026-07-08 14:05:00 EST"
    assert first.tools.db is first.user_tools.db
    assert not second.tools.db.task_config.data


def test_machine_readable_tool_contract_matches_implemented_business_tools() -> None:
    environment = get_environment(retrieval_variant="bm25")
    payload = json.loads(
        (SYNTHETIC_BANKING_DATA_DIR / "tool_registry.json").read_text(encoding="utf-8")
    )
    contracts = {tool["name"] for tool in payload["tools"]}
    agent_tools = set(environment.tools.tools)
    user_tools = set(environment.user_tools.tools)

    assert {
        "file_credit_card_transaction_dispute_7383",
        "file_debit_card_transaction_dispute_5724",
        "open_bank_account_5464",
        "close_bank_account_1097",
        "open_business_deposit_account_8514",
        "submit_customer_transfer_2674",
    } <= contracts & agent_tools
    assert {
        "get_card_last_4_digits",
        "get_debit_card_last_4_digits",
        "provide_written_dispute_statement",
        "apply_for_credit_card",
        "submit_referral",
    } <= contracts & user_tools
    assert "record_debit_dispute_intake_outcome" in contracts & agent_tools
