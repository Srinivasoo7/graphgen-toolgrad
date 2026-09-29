"""The kept ITSM row must not count as a verified training row."""

from bridge.chain_verifier import MockToolExecutor, verify_qa
from bridge.datasets.release import ReleasePolicy, assess_release
from bridge.verification.intent import intent_aligned


def test_status_question_on_create_ticket_is_not_aligned():
    assert intent_aligned(
        "What is the status of INC-1042?",
        ["create_ticket"],
    ) is False


def test_create_question_on_create_ticket_continues():
    assert intent_aligned("Create a ticket for the laptop", ["create_ticket"]) is True


def test_close_question_on_read_is_rejected():
    assert intent_aligned("Close ticket INC-1042", ["get_ticket"]) is False


def test_execution_success_is_not_safe_to_review():
    executor = MockToolExecutor()
    executor.register("create_ticket", {"type": "object", "properties": {}}, lambda ti: {"status": "Open"})
    report = verify_qa(
        {
            "question": "What is the status of INC-1042?",
            "answer_draft": "Open",
            "chain": [{
                "tool": "create_ticket",
                "tool_input": {},
                "result_preview": '{"status": "Open"}',
            }],
        },
        executor,
    )
    assert report["execution_completed"] is True
    assert report["chain_valid"] is True
    assert report["intent_aligned"] is False
    assert report["safe_to_review"] is False


def test_release_without_expert_acceptance_is_not_released():
    verdict = assess_release(
        accepted_ids=[],
        train_ids=[],
        validation_ids=[],
        expert_decisions=0,
        outcome_assertions=False,
        policy=ReleasePolicy(),
    )
    assert verdict["status"] != "released"
    assert "no expert decisions" in verdict["reasons"]
