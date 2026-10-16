"""Types for prompts written as a conversation.

A *chat* is the conversation you want a model to answer: a list of messages,
each a dictionary with a ``role`` and its text ``content``. The model's answer
is measured right after the last message:

- If the last message is from the user, the answer starts a new assistant turn.
- If the last message is from the assistant, its content is a *prefill*: text
  the answer continues from, such as ``"Answer:"``. It is kept exactly as
  written, including any trailing spaces or newlines.

Earlier assistant messages are ordinary completed turns, for example worked
examples in a few-shot prompt.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

type Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """One message in a chat.

    Attributes:
        role: Who wrote the message: ``"system"``, ``"user"``, or ``"assistant"``.
        content: The message text, exactly as it should appear to the model.
    """

    role: Role
    content: str


type ChatSpec = Sequence[Message]
"""A chat to measure: messages in order, ending with a user or assistant message."""

_ROLES: dict[str, Role] = {"system": "system", "user": "user", "assistant": "assistant"}


def validate_chat(messages: Sequence[Mapping[str, object]]) -> list[Message]:
    """Check that a chat has a shape the model can answer, and return a copy.

    A valid chat has at least one message. Each message is a dictionary with
    exactly the keys ``role`` and ``content``, a known role, and text content.
    The last message must be from the user or the assistant, because the
    answer is measured right after it. Content is never changed: an empty or
    whitespace-ending assistant prefill is kept as it is.

    This checks structure only. Whether a model's chat template accepts a
    particular order of roles is checked when the chat is rendered.

    Args:
        messages: The chat, as a list of message dictionaries.

    Returns:
        A new list of new message dictionaries, so later changes to
        ``messages`` cannot affect a measurement that uses the copy.

    Raises:
        TypeError: ``messages`` is a string, a single message dictionary, or
            not a sequence; a message is not a dictionary; or content is not
            a string.
        ValueError: The chat is empty, a message has missing or extra keys or
            an unknown role, or the last message is a system message.

    Examples:
        >>> chat = validate_chat(
        ...     [
        ...         {"role": "user", "content": "Is the sky blue? Answer yes or no."},
        ...         {"role": "assistant", "content": "Answer: "},
        ...     ]
        ... )
        >>> chat[-1]["content"]
        'Answer: '
        >>> validate_chat([{"role": "system", "content": "Be brief."}])
        Traceback (most recent call last):
            ...
        ValueError: The last message must be from the user or the assistant, not 'system'
    """
    # A string is a Sequence and a message is a Mapping; without these checks
    # a prompt string or a lone message would be misread message by message.
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, Sequence):
        raise TypeError("A chat must be a list of message dictionaries")
    if not messages:
        raise ValueError("A chat needs at least one message")
    chat = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"Message {index} must be a dictionary with 'role' and 'content'")
        keys = set(message)
        if keys != {"role", "content"}:
            found = sorted(map(repr, keys))
            raise ValueError(
                f"Message {index} must have exactly the keys 'role' and 'content'; "
                f"got {', '.join(found) or 'none'}"
            )
        role = message["role"]
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError(
                f"Message {index} has role {role!r}; expected 'system', 'user', or 'assistant'"
            )
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError(f"Message {index} content must be a string; got {content!r}")
        chat.append(Message(role=_ROLES[role], content=content))
    if chat[-1]["role"] == "system":
        raise ValueError("The last message must be from the user or the assistant, not 'system'")
    return chat
