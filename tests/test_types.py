"""Chats are checked for shape without ever changing what they say."""

import pytest

from credences.types import validate_chat

USER = {"role": "user", "content": "Is the sky blue? Answer yes or no."}


def test_prompt_string_is_not_read_as_a_chat_of_characters():
    with pytest.raises(TypeError, match="list of message"):
        validate_chat("Is the sky blue?")


def test_lone_message_must_be_wrapped_in_a_list():
    with pytest.raises(TypeError, match="list of message"):
        validate_chat(USER)


def test_empty_chat_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        validate_chat([])


def test_answer_cannot_be_measured_after_a_final_system_message():
    with pytest.raises(ValueError, match="last message"):
        validate_chat([USER, {"role": "system", "content": "Be brief."}])


@pytest.mark.parametrize(
    ("message", "error", "match"),
    [
        ({"role": "user"}, ValueError, "exactly the keys"),
        ({**USER, "name": "Ada"}, ValueError, "exactly the keys"),
        ({"role": "tool", "content": "42"}, ValueError, "role 'tool'"),
        ({"role": None, "content": "hi"}, ValueError, "role None"),
        ({"role": "user", "content": 42}, TypeError, "content must be a string"),
        (("user", "hi"), TypeError, "must be a dictionary"),
    ],
)
def test_malformed_messages_are_rejected_before_rendering(message, error, match):
    with pytest.raises(error, match=match):
        validate_chat([message])


@pytest.mark.parametrize("prefill", ["", "Answer:", "Answer: ", "Answer:\n\n"])
def test_assistant_prefill_is_kept_exactly_including_trailing_whitespace(prefill):
    chat = validate_chat([USER, {"role": "assistant", "content": prefill}])

    assert chat[-1] == {"role": "assistant", "content": prefill}


def test_completed_assistant_turns_are_allowed_as_examples():
    chat = [
        {"role": "system", "content": "Answer with one word."},
        {"role": "user", "content": "Is snow cold?"},
        {"role": "assistant", "content": "yes"},
        USER,
    ]

    assert validate_chat(chat) == chat


def test_returned_chat_is_unaffected_by_later_edits_to_the_input():
    original = [dict(USER)]
    chat = validate_chat(original)
    original[0]["content"] = "changed"
    original.append({"role": "assistant", "content": "no"})

    assert chat == [USER]
