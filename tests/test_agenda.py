from datetime import UTC, datetime

from research_lab.agenda import (
    RESEARCH_AGENDA,
    half_hour_window,
    scheduled_client_question,
    scheduled_question,
)


def test_half_hour_window_is_stable_and_bounded():
    assert half_hour_window(datetime(2026, 7, 27, 4, 29, 59, tzinfo=UTC)).endswith("04:00:00+00:00")
    assert half_hour_window(datetime(2026, 7, 27, 4, 30, 1, tzinfo=UTC)).endswith("04:30:00+00:00")


def test_agenda_rotates_each_half_hour_and_wraps():
    start = datetime(2026, 7, 27, 0, 0, tzinfo=UTC)
    next_slot = datetime(2026, 7, 27, 0, 30, tzinfo=UTC)
    wrapped = datetime(2026, 7, 27, 4, 0, tzinfo=UTC)
    assert scheduled_question(start) != scheduled_question(next_slot)
    assert scheduled_question(start) == scheduled_question(wrapped)
    assert scheduled_question(start) in RESEARCH_AGENDA


def test_active_client_questions_take_priority():
    instant = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
    questions = ["client membrane objective", "client alloy objective"]
    assert scheduled_client_question(questions, instant) in questions
    assert scheduled_client_question([], instant) == scheduled_question(instant)
