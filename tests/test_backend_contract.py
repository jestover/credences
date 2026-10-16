"""Backends must report every requested score; truncated readouts are rejected."""

import math

import pytest
from fakes import DeclaredLogitsBackend, VocabularyTokenizer, score_with_backend

from credences.errors import BackendReadoutError
from credences.trie import CandidatePaths

TOKENIZER = VocabularyTokenizer(["Is it good?", "pos", "itive", "ition", "e", " pos"])
POS, ITIVE, ITION, E, SPACE_POS = (TOKENIZER.ids[p] for p in ("pos", "itive", "ition", "e", " pos"))
CONTEXT = tuple(
    TOKENIZER.apply_chat_template(
        [{"role": "user", "content": "Is it good?"}],
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
    )
)
# All three answers share a first token, so after it the model must choose
# among three continuations.
FORMS = {
    label: CandidatePaths(
        TOKENIZER.encode(label, add_special_tokens=False),
        TOKENIZER.encode(" " + label, add_special_tokens=False),
    )
    for label in ("positive", "position", "pose")
}
TABLES = {
    CONTEXT: {POS: 1.0, SPACE_POS: 0.0},
    (*CONTEXT, POS): {ITIVE: math.log(0.5), ITION: math.log(0.3), E: math.log(0.2)},
}


def test_complete_readouts_score_the_declared_example():
    backend = DeclaredLogitsBackend(TOKENIZER, TABLES)
    selected, logs = score_with_backend(backend, CONTEXT, FORMS)

    assert selected == {
        "positive": (POS, ITIVE),
        "position": (POS, ITION),
        "pose": (POS, E),
    }
    assert {label: math.exp(value) for label, value in logs.items()} == pytest.approx(
        {"positive": 0.5, "position": 0.3, "pose": 0.2}
    )
    assert backend.calls == [
        (CONTEXT, (POS, SPACE_POS)),
        ((*CONTEXT, POS), (ITIVE, ITION, E)),
    ]


# top_k=1 drops a first-token score; top_k=2 drops a later continuation score.
@pytest.mark.parametrize(("top_k", "calls_before_error"), [(1, 1), (2, 2)])
def test_backend_returning_only_its_top_scores_is_rejected_not_renormalized(
    top_k, calls_before_error
):
    backend = DeclaredLogitsBackend(TOKENIZER, TABLES, top_k=top_k)

    with pytest.raises(BackendReadoutError, match="Missing requested score"):
        score_with_backend(backend, CONTEXT, FORMS)
    assert len(backend.calls) == calls_before_error
