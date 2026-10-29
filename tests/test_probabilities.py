"""Masked probability rules, checked against an independent summed-spelling oracle."""

import math
from decimal import Decimal, localcontext

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from credences.errors import BackendReadoutError
from credences.probabilities import (
    RawReadout,
    ScoredPath,
    TokenScores,
    coverage,
    logprob_sum,
    logsumexp,
    masked_logprobs,
    measurement_from_raw,
    validate_readout,
)
from credences.trie import CandidatePaths, build_trie

# Both spellings are explicit fixtures, not tokenizer assumptions. The second
# set has spellings that start differently; the third shares first tokens across
# candidates; the fourth needs a grouped stopping event; the fifth has a
# candidate whose two spellings are the same token path. END ids are outside all
# candidate paths.
FORM_SETS = [
    {"A": ((0,), (1,)), "B": ((2,), (3,))},
    {"positive": ((0, 1), (4,)), "position": ((0, 2, 3), (5,))},
    {
        "very negative": ((0, 1), (6, 1)),
        "negative": ((2,), (7,)),
        "neutral": ((3,), (8,)),
        "positive": ((4,), (9,)),
        "very positive": ((0, 5), (6, 5)),
    },
    {"A": ((0,), (3,)), "AB": ((0, 1), (3, 1))},
    {"A": ((0,), (0,)), "B": ((1, 2), (3,))},
]
END = (10, 11)
LOGIT = st.one_of(
    st.just(-math.inf),
    st.floats(min_value=-80, max_value=80, width=32, allow_nan=False, allow_infinity=False),
)


def _strict_prefix(short, long):
    return len(short) < len(long) and long[: len(short)] == short


@st.composite
def generated_forms(draw):
    count = draw(st.integers(min_value=2, max_value=6))
    path = st.lists(st.integers(0, 9), min_size=1, max_size=5).map(tuple)
    paths = draw(st.lists(path, min_size=2 * count, max_size=2 * count, unique=True))
    forms = {}
    for index in range(count):
        bare, spaced = paths[2 * index], paths[2 * index + 1]
        if draw(st.booleans()) and draw(st.booleans()):
            spaced = bare  # Some tokenizers give both spellings one path.
        assume(not _strict_prefix(bare, spaced) and not _strict_prefix(spaced, bare))
        forms[str(index)] = (bare, spaced)
    return forms


@st.composite
def summed_spelling_cases(draw):
    forms = draw(st.one_of(st.sampled_from(FORM_SETS), generated_forms()))
    paths = [path for spellings in forms.values() for path in spellings]
    prefixes = sorted({path[:i] for path in paths for i in range(len(path))})
    logits = {}
    for prefix in prefixes:
        tokens = sorted(
            {
                path[len(prefix)]
                for path in paths
                if path[: len(prefix)] == prefix and len(path) > len(prefix)
            }
            | set(END)
        )
        values = draw(st.lists(LOGIT, min_size=len(tokens), max_size=len(tokens)))
        logits[prefix] = dict(zip(tokens, values, strict=True))
    return forms, logits


def _token_scores(table):
    """Declared logits plus a vocabulary normalizer that also covers one unlisted token."""
    finite = [value for value in table.values() if value != -math.inf]
    total = logsumexp(finite) if finite else -math.inf
    return TokenScores(logits=table, log_normalizer=logsumexp([total, 0.0]))


def _naive_reference(forms, logits):
    """Enumerate every distinct path and multiply Decimal probabilities along it.

    Scan complete paths at each step: no shared trie/plan or production math.
    Direct exponentials avoid copying the production log-softmax algorithm.
    """
    paths = [
        (label, path) for label, spellings in forms.items() for path in dict.fromkeys(spellings)
    ]
    with localcontext() as context:
        context.prec = 80
        path_probabilities = []
        for label, path in paths:
            probability = Decimal(1)
            for depth in range(len(path) + 1):
                prefix = path[:depth]
                children = {
                    other[depth]
                    for _, other in paths
                    if len(other) > depth and other[:depth] == prefix
                }
                if not children:
                    break
                endings = set(END) if any(other == prefix for _, other in paths) else set()
                allowed = children | endings
                if len(allowed) == 1:
                    transitions = {next(iter(allowed)): Decimal(1)}
                else:
                    weights = {
                        token: Decimal(0)
                        if logits[prefix][token] == -math.inf
                        else Decimal.from_float(logits[prefix][token]).exp()
                        for token in allowed
                    }
                    denominator = sum(weights.values())
                    transitions = {
                        token: weight / denominator if denominator else Decimal(1) / len(allowed)
                        for token, weight in weights.items()
                    }
                if depth == len(path):
                    probability *= sum(transitions[token] for token in endings)
                else:
                    probability *= transitions[path[depth]]
            path_probabilities.append((label, path, probability))
        credences = {}
        for label, _, probability in path_probabilities:
            credences[label] = credences.get(label, Decimal(0)) + probability

        def log(value):
            return float(value.ln()) if value else -math.inf

        return (
            {(label, path): log(p) for label, path, p in path_probabilities},
            {label: log(value) for label, value in credences.items()},
        )


def _score_trie(forms, logits):
    """Drive the production plan using declared scores, not a model."""
    prepared = {
        label: CandidatePaths(bare=bare, spaced=spaced) for label, (bare, spaced) in forms.items()
    }
    plan = build_trie(prepared, end_tokens=END)
    scores = {request.prefix: _token_scores(logits[request.prefix]) for request in plan.readouts}
    return plan.score(scores, context_token_count=0)


@given(case=summed_spelling_cases())
def test_summed_spellings_match_independent_reference(case):
    forms, logits = case
    expected_paths, expected_credences = _naive_reference(forms, logits)
    raw = _score_trie(forms, logits)

    assert raw.credence_logprobs == pytest.approx(expected_credences, abs=1e-12)
    scored = {
        (label, scored_path.token_ids): scored_path.logprob
        for label, paths in raw.scored_paths.items()
        for scored_path in paths
    }
    assert scored == pytest.approx(expected_paths, abs=1e-12)
    total = math.fsum(math.exp(value) for value in raw.credence_logprobs.values())
    assert total == pytest.approx(1.0, abs=1e-12)
    assert math.isfinite(measurement_from_raw(raw).entropy)
    reversed_raw = _score_trie(dict(reversed(list(forms.items()))), logits)
    assert list(reversed_raw.credence_logprobs) == list(reversed(list(raw.credence_logprobs)))
    assert reversed_raw.credence_logprobs == pytest.approx(raw.credence_logprobs, abs=1e-12)


def test_keeping_only_the_stronger_spelling_would_fail_the_reference():
    # The reference is sensitive to the old rule: dropping the weaker spelling
    # changes every credence that it carried mass for.
    forms = FORM_SETS[0]
    logits = {(): {0: 2.0, 1: 0.0, 2: 1.0, 3: 0.5, 10: 0.0, 11: 0.0}}
    _, summed = _naive_reference(forms, logits)
    stronger_only = masked_logprobs([0, 2], logits[()])

    assert summed["A"] != pytest.approx(stronger_only[0], abs=1e-3)
    assert _score_trie(forms, logits).credence_logprobs == pytest.approx(summed, abs=1e-12)


def test_forced_transition_does_not_even_inspect_the_readout():
    class Unqueried:
        def __getitem__(self, token):
            pytest.fail("A forced transition must not request a score")

    assert masked_logprobs([4, 4], Unqueried()) == {4: 0.0}


def test_uniform_fallback_counts_distinct_tokens_and_groups_end_mass():
    logs = masked_logprobs([1, 10, 11, 10], {1: -math.inf, 10: -math.inf, 11: -math.inf})

    assert logs == {1: -math.log(3), 10: -math.log(3), 11: -math.log(3)}
    assert math.exp(logprob_sum(logs[token] for token in END)) == pytest.approx(2 / 3)


def test_token_uniform_fallback_is_not_label_uniform():
    logits = {
        (): {token: -math.inf for token in range(10)},
        (0,): {1: -math.inf, 5: -math.inf},
        (6,): {1: -math.inf, 5: -math.inf},
    }
    raw = _score_trie(FORM_SETS[2], logits)

    assert {label: math.exp(value) for label, value in raw.credence_logprobs.items()} == (
        pytest.approx(
            {
                "very negative": 1 / 8,
                "negative": 1 / 4,
                "neutral": 1 / 4,
                "positive": 1 / 4,
                "very positive": 1 / 8,
            }
        )
    )


@pytest.mark.parametrize("offset", [0.0, -1e30, 1e30])
def test_large_common_logit_offset_does_not_destroy_equal_weights(offset):
    assert masked_logprobs([1, 2], {1: offset, 2: offset}) == {1: -math.log(2), 2: -math.log(2)}


def test_finite_float32_extremes_keep_finite_logs_through_underflow():
    logs = masked_logprobs([1, 2, 3], {1: 3e38, 2: -3e38, 3: -math.inf})

    assert logs[1] == 0.0
    assert logs[2] == -6e38
    assert math.exp(logs[2]) == 0.0
    assert logs[3] == -math.inf


def test_tiny_finite_scores_never_trigger_uniform_fallback():
    logs = masked_logprobs([1, 2, 3], {1: -1001.0, 2: -1000.0, 3: -math.inf})

    assert math.exp(logs[2]) == pytest.approx(1 / (1 + math.exp(-1)))
    assert logs[3] == -math.inf


def test_readout_validation_requires_every_requested_score_even_when_other_scores_exist():
    with pytest.raises(BackendReadoutError, match="2"):
        masked_logprobs([1, 2], {1: 0.0, 3: 0.0})


@pytest.mark.parametrize("score", [math.nan, math.inf, "0", None, True])
def test_malformed_requested_scores_are_errors_not_uniform_fallback(score):
    with pytest.raises(BackendReadoutError, match="2"):
        masked_logprobs([1, 2], {1: -math.inf, 2: score})


def test_readout_validation_covers_every_first_id_and_ignores_unrequested_extras():
    assert validate_readout([1, 1, 2], {1: 0.0, 2: -math.inf, 99: math.nan}) == {
        1: 0.0,
        2: -math.inf,
    }


def test_empty_allowed_set_is_a_structural_error():
    with pytest.raises(ValueError, match="nonempty"):
        masked_logprobs([], {})


def test_logsumexp_handles_empty_and_zero_mass_without_nan():
    assert logsumexp([]) == -math.inf
    assert logsumexp([-math.inf, -math.inf]) == -math.inf
    assert logsumexp([-1000.0, -1001.0, -math.inf]) == pytest.approx(
        -1000.0 + math.log1p(math.exp(-1))
    )


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_logsumexp_rejects_invalid_numbers_instead_of_hiding_them(value):
    with pytest.raises(ValueError):
        logsumexp([0.0, value])


def test_tiny_finite_tail_mass_is_retained_in_log_normalization():
    correction = math.log1p(math.exp(-40))
    logs = masked_logprobs([1, 2], {1: 0.0, 2: -40.0})

    assert logs[1] == -correction
    assert logs[1] < 0
    assert logsumexp([0.0, -40.0]) == correction


def test_zero_incoming_mass_stays_zero_after_uniform_fallback():
    root = masked_logprobs([1, 2], {1: -math.inf, 2: 0.0})
    branch = masked_logprobs([3, 4], {3: -math.inf, 4: -math.inf})

    assert root[1] + branch[3] == -math.inf
    assert root[1] + branch[4] == -math.inf


def test_positive_position_golden_example_sums_both_spellings():
    # p(" positive") = 0.40, p("pos") = 0.30, p(" position") = 0.20 at the root;
    # after "pos", "itive" gets 0.6 and "it" gets 0.4.
    logits = {
        (): {0: math.log(0.30), 4: math.log(0.40), 5: math.log(0.20)},
        (0,): {1: math.log(0.6), 2: math.log(0.4)},
    }
    raw = _score_trie(FORM_SETS[1], logits)

    assert {label: math.exp(value) for label, value in raw.credence_logprobs.items()} == (
        pytest.approx(
            {"positive": (0.40 + 0.30 * 0.6) / 0.9, "position": (0.20 + 0.30 * 0.4) / 0.9}
        )
    )


@pytest.mark.parametrize("continuation", [-math.inf, -80.0, -40.0, -1000.0])
def test_grouped_end_roundoff_cannot_make_a_valid_measurement_fail(continuation):
    steps = masked_logprobs([1, 10, 11], {1: continuation, 10: -4.5, 11: -0.5})
    end_log = logprob_sum(steps[token] for token in END)
    raw = RawReadout(
        credence_logprobs={"A": end_log, "AB": steps[1]},
        scored_paths={
            "A": (ScoredPath(("bare",), (0,), end_log),),
            "AB": (ScoredPath(("bare",), (0, 1), steps[1]),),
        },
        context_token_count=0,
        coverage=1.0,
    )
    result = measurement_from_raw(raw)

    assert end_log <= 0.0
    assert end_log == pytest.approx(0.0, abs=1e-16)
    assert result.credences["A"] <= 1.0
    assert result.top_labels == ("A",)
    if math.isfinite(continuation):
        assert math.isfinite(result.raw.credence_logprobs["AB"])
    else:
        assert result.raw.credence_logprobs["AB"] == -math.inf


def test_grouped_event_preserves_tiny_negative_logs_and_zero_mass():
    assert logprob_sum([-1e-20]) == -1e-20
    assert logprob_sum([]) == -math.inf
    assert logprob_sum([-math.inf, -math.inf]) == -math.inf


def test_grouped_event_rejects_real_excess_mass_instead_of_clamping_it():
    with pytest.raises(ValueError, match="exceeds 1"):
        logprob_sum([math.log(0.8), math.log(0.8)])
    with pytest.raises(ValueError, match="positive"):
        logprob_sum([0.1])
    # Positive output is legitimate for generic weights, unlike probabilities.
    assert logsumexp([0.0, 0.0]) == math.log(2)


def test_coverage_is_the_full_vocabulary_probability_of_the_allowed_tokens():
    # Allowed tokens have probabilities 0.07 and 0.02; "The" takes 0.90 elsewhere.
    scores = TokenScores(
        logits={1: math.log(0.07), 2: math.log(0.02)},
        log_normalizer=math.log(0.07 + 0.02 + 0.90 + 0.01),
    )

    assert coverage([1, 2], scores) == pytest.approx(0.09)
    assert coverage([1, 2], TokenScores({1: -math.inf, 2: -math.inf}, 0.0)) == 0.0


def test_coverage_never_changes_credences():
    # The coverage section's example: a model that opens with "The" and one that
    # answers directly have the same credences.
    forms = {"yes": CandidatePaths((1,), (3,)), "no": CandidatePaths((2,), (4,))}
    plan = build_trie(forms)
    allowed = {1: 0.07, 2: 0.02, 3: 0.0, 4: 0.0}
    logits = {token: math.log(p) if p else -math.inf for token, p in allowed.items()}
    habitual = plan.score({(): TokenScores(logits, math.log(1.0))}, context_token_count=5)
    direct = plan.score({(): TokenScores(logits, math.log(0.1))}, context_token_count=5)

    assert habitual.credence_logprobs == direct.credence_logprobs
    assert habitual.coverage == pytest.approx(0.09)
    assert direct.coverage == pytest.approx(0.9)


@pytest.mark.parametrize(
    "log_normalizer",
    [math.nan, math.inf, -math.inf, math.log(0.05), "0"],
    ids=["nan", "inf", "-inf", "below-requested-mass", "text"],
)
def test_invalid_log_normalizer_is_a_backend_error(log_normalizer):
    scores = TokenScores({1: math.log(0.07), 2: math.log(0.02)}, log_normalizer)

    with pytest.raises(BackendReadoutError, match="normalizer"):
        coverage([1, 2], scores)
