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
# All three answers share a first token in each spelling, so after "pos" and
# after " pos" the model must choose among three continuations.
FORMS = {
    label: CandidatePaths(
        TOKENIZER.encode(label, add_special_tokens=False),
        TOKENIZER.encode(" " + label, add_special_tokens=False),
    )
    for label in ("positive", "position", "pose")
}
CONTINUATIONS = {ITIVE: math.log(0.5), ITION: math.log(0.3), E: math.log(0.2)}
TABLES = {
    CONTEXT: {POS: 1.0, SPACE_POS: 0.0},
    (*CONTEXT, POS): CONTINUATIONS,
    (*CONTEXT, SPACE_POS): CONTINUATIONS,
}


def _credences(raw):
    return {label: math.exp(value) for label, value in raw.credence_logprobs.items()}


def test_complete_readouts_score_both_spellings_of_the_declared_example():
    backend = DeclaredLogitsBackend(TOKENIZER, TABLES)
    plan, raw = score_with_backend(backend, CONTEXT, FORMS)

    assert [path.token_ids for path in raw.scored_paths["positive"]] == [
        (POS, ITIVE),
        (SPACE_POS, ITIVE),
    ]
    assert _credences(raw) == pytest.approx({"positive": 0.5, "position": 0.3, "pose": 0.2})
    assert backend.calls == [
        (CONTEXT, (POS, SPACE_POS)),
        ((*CONTEXT, POS), (ITIVE, ITION, E)),
        ((*CONTEXT, SPACE_POS), (ITIVE, ITION, E)),
    ]
    assert raw.context_token_count == len(CONTEXT)


def test_coverage_reflects_the_rest_of_the_vocabulary_without_changing_credences():
    # Giving every other vocabulary token logit 0 makes answering less likely
    # overall, but the allowed tokens' scores, and so the credences, are unchanged.
    _, focused = score_with_backend(DeclaredLogitsBackend(TOKENIZER, TABLES), CONTEXT, FORMS)
    _, distracted = score_with_backend(
        DeclaredLogitsBackend(TOKENIZER, TABLES, fill=0.0), CONTEXT, FORMS
    )
    other_tokens = len(TOKENIZER.pieces) - 2

    assert focused.coverage == pytest.approx(1.0)
    assert distracted.coverage == pytest.approx((math.e + 1) / (math.e + 1 + other_tokens))
    assert distracted.credence_logprobs == focused.credence_logprobs


# top_k=1 drops the weaker first token; top_k=2 drops the weakest continuation.
@pytest.mark.parametrize(("top_k", "dropped"), [(1, SPACE_POS), (2, E)])
def test_backend_returning_only_its_top_scores_is_rejected_not_renormalized(top_k, dropped):
    backend = DeclaredLogitsBackend(TOKENIZER, TABLES, top_k=top_k)

    with pytest.raises(BackendReadoutError, match=f"Missing requested score for token {dropped}$"):
        score_with_backend(backend, CONTEXT, FORMS)
