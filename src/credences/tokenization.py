"""The tokenizer interface credences uses to turn text into token IDs.

Every model backend supplies a tokenizer that matches its model. This module
defines the methods credences needs from it. Hugging Face tokenizers, which
most local models ship with, already provide them.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol


class Tokenizer(Protocol):
    """The tokenizer methods and attributes credences relies on.

    Tokenizer libraries disagree on defaults. For example, Hugging Face
    tokenizers add special tokens such as a beginning-of-text marker unless
    told not to, which would silently put an extra token in front of an
    answer. So every option here has no default: callers always say exactly
    what they want, and any tokenizer whose methods accept these keyword
    arguments fits, whatever its own defaults are.

    Attributes:
        eos_token_id: The token ID that ends a sequence, if the tokenizer has one.
        chat_template: The template that formats a chat as model input, or
            None for a base model without one.
        additional_special_tokens: Extra control tokens the model uses, such
            as markers for reasoning sections.
    """

    eos_token_id: int | None
    chat_template: str | None
    additional_special_tokens: list[str]

    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]:
        """Return the token IDs for ``text``.

        Args:
            text: The text to encode.
            add_special_tokens: Whether to add markers such as a
                beginning-of-text token around the text.

        Returns:
            The token IDs, in order.
        """
        ...

    def decode(
        self,
        ids: Sequence[int],
        /,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """Return the text for a sequence of token IDs.

        Args:
            ids: The token IDs to decode.
            skip_special_tokens: Whether to leave control tokens out of the text.
            clean_up_tokenization_spaces: Whether to let the tokenizer adjust
                spaces around punctuation. credences passes False so decoded
                text can be compared exactly.

        Returns:
            The decoded text.
        """
        ...

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        /,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        continue_final_message: bool,
        **chat_template_kwargs: object,
    ) -> str | list[int]:
        """Format a chat as the model's input, using the model's own template.

        Args:
            messages: The chat, as role and content dictionaries.
            tokenize: Whether to return token IDs instead of text.
            add_generation_prompt: Whether to open a new assistant turn after
                the last message.
            continue_final_message: Whether to leave the last message open so
                the model continues it, as for an assistant prefill. Not used
                together with ``add_generation_prompt``.
            **chat_template_kwargs: Model-specific template options, such as a
                switch that turns off reasoning output.

        Returns:
            The formatted text, or its token IDs when ``tokenize`` is True.
        """
        ...
