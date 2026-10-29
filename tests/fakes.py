"""Small stand-ins for a tokenizer and model backend, with declared behavior.

Tests use these to drive credences through the same interfaces a real model
backend implements, without model weights. ``uv run ty check`` includes this
file, so a fake that drifts from the real interfaces fails type checking.
"""

import math
from collections.abc import Mapping, Sequence

from credences.backends import FullLogitsBackend, TokenScores
from credences.probabilities import RawReadout, logsumexp
from credences.tokenization import Tokenizer
from credences.trie import CandidatePaths, TriePlan, build_trie

SPECIAL_PIECES = ("<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")


class VocabularyTokenizer:
    """Encode text by greedy longest match over a small, closed vocabulary.

    Token IDs are positions in ``pieces``. Text that the vocabulary cannot
    cover raises ValueError, so a test cannot silently tokenize something it
    did not declare. The chat template wraps each message as
    ``<|role|>content<|end|>``.
    """

    def __init__(self, pieces: Sequence[str]) -> None:
        self.pieces = (*SPECIAL_PIECES, *pieces)
        if len(set(self.pieces)) != len(self.pieces):
            raise ValueError("Vocabulary pieces must be distinct")
        self.ids = {piece: index for index, piece in enumerate(self.pieces)}
        self.eos_token_id: int | None = self.ids["<|end|>"]
        self.chat_template: str | None = "<|role|>content<|end|>"
        self.additional_special_tokens: list[str] = list(SPECIAL_PIECES)

    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]:
        ids = [self.ids["<|bos|>"]] if add_special_tokens else []
        position = 0
        while position < len(text):
            piece = max(
                (piece for piece in self.pieces if text.startswith(piece, position)),
                key=len,
                default=None,
            )
            if piece is None:
                raise ValueError(f"No vocabulary piece matches {text[position:]!r}")
            ids.append(self.ids[piece])
            position += len(piece)
        return ids

    def decode(
        self,
        ids: Sequence[int],
        /,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        pieces = (self.pieces[token] for token in ids)
        if skip_special_tokens:
            pieces = (piece for piece in pieces if piece not in SPECIAL_PIECES)
        return "".join(pieces)

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
        if add_generation_prompt and continue_final_message:
            raise ValueError("Cannot open a new assistant turn and continue the last message")
        text = ""
        for index, message in enumerate(messages):
            text += f"<|{message['role']}|>{message['content']}"
            if not (continue_final_message and index == len(messages) - 1):
                text += "<|end|>"
        if add_generation_prompt:
            text += "<|assistant|>"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class DeclaredLogitsBackend:
    """Return declared logits at declared token contexts, and record every call.

    ``tables`` maps a complete input (prompt plus answer tokens) to logits for
    that next position. Every vocabulary token without a declared logit gets
    ``fill``, both when requested and in the full-vocabulary normalizer.
    Asking about an undeclared context raises LookupError, because it means the
    test reached a state it did not plan for.

    ``top_k`` imitates a faulty backend that returns only its highest-scoring
    requested tokens instead of all of them.
    """

    def __init__(
        self,
        tokenizer: VocabularyTokenizer,
        tables: Mapping[Sequence[int], Mapping[int, float]],
        *,
        fill: float = -math.inf,
        top_k: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.tables = {tuple(context): dict(scores) for context, scores in tables.items()}
        self.fill = fill
        self.top_k = top_k
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def tokenizer(self) -> VocabularyTokenizer:
        return self._tokenizer

    def next_token_logits(self, token_ids: Sequence[int], wanted: Sequence[int]) -> TokenScores:
        context = tuple(token_ids)
        self.calls.append((context, tuple(wanted)))
        if context not in self.tables:
            raise LookupError(f"No logits declared after {context}")
        table = self.tables[context]
        vocabulary = range(len(self._tokenizer.pieces))
        log_normalizer = logsumexp(table.get(token, self.fill) for token in vocabulary)
        scores = {token: table.get(token, self.fill) for token in wanted}
        if self.top_k is not None:
            kept = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_k]
            scores = {token: scores[token] for token in kept}
        return TokenScores(logits=scores, log_normalizer=log_normalizer)


def score_with_backend(
    backend: FullLogitsBackend,
    context: Sequence[int],
    forms: Mapping[str, CandidatePaths],
    *,
    end_tokens: Sequence[int] = (),
) -> tuple[TriePlan, RawReadout]:
    """Run the scoring steps against a backend, as a measurement will.

    ``context`` holds the prompt tokens up to where scoring starts. Returns the
    plan and the scored result.
    """
    plan = build_trie(forms, end_tokens=end_tokens)
    scores = {
        request.prefix: backend.next_token_logits((*context, *request.prefix), request.wanted)
        for request in plan.readouts
    }
    return plan, plan.score(scores, context_token_count=len(context))


def _conforms(
    backend: DeclaredLogitsBackend, tokenizer: VocabularyTokenizer
) -> tuple[FullLogitsBackend, Tokenizer]:
    """Fail type checking if a fake stops matching the interface it stands in for."""
    return backend, tokenizer
