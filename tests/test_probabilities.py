"""Masked probability rules, checked against an independent selected-path oracle."""

import math
from decimal import Decimal, localcontext

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credences import RawReadout
from credences.errors import BackendReadoutError
from credences.probabilities import (
    logprob_sum,
    logsumexp,
    masked_logprobs,
    measurement_from_raw,
    validate_readout,
)
from credences.trie import (
    CandidatePaths,
    build_trie,
    first_token_ids,
    select_paths,
    validate_potential_paths,
)

# Both forms are explicit fixtures, not tokenizer assumptions. The second
# changes branch points with form selection; the last needs a grouped stopping
# event. END ids are outside all candidate paths.
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
]
END = (10, 11)
LOGIT = st.one_of(
    st.just(-math.inf),
    st.floats(min_value=-80, max_value=80, width=32, allow_nan=False, allow_infinity=False),
)


@st.composite
def generated_forms(draw):
    count = draw(st.integers(min_value=2, max_value=6))
    path = st.lists(st.integers(0, 9), min_size=1, max_size=5).map(tuple)
    paths = draw(st.lists(path, min_size=2 * count, max_size=2 * count, unique=True))
    return {str(index): (paths[2 * index], paths[2 * index + 1]) for index in range(count)}


@st.composite
def selected_path_cases(draw):
    forms = draw(st.one_of(st.sampled_from(FORM_SETS), generated_forms()))
    paths = [path for alternatives in forms.values() for path in alternatives]
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


def _naive_reference(forms, logits):
    """Select independently, then multiply Decimal probabilities per candidate.

    Scan complete paths at each step: no shared trie/plan or production math.
    Direct exponentials avoid copying the production log-softmax algorithm.
    """
    selected = {}
    for label, (bare, spaced) in forms.items():
        selected[label] = bare if logits[()][bare[0]] >= logits[()][spaced[0]] else spaced

    with localcontext() as context:
        context.prec = 80
        result = {}
        for label, path in selected.items():
            probability = Decimal(1)
            for depth in range(len(path) + 1):
                prefix = path[:depth]
                children = {
                    other[depth]
                    for other in selected.values()
                    if len(other) > depth and other[:depth] == prefix
                }
                if not children:
                    break
                endings = set(END) if prefix in selected.values() else set()
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
            result[label] = float(probability.ln()) if probability else -math.inf
    return selected, result


def _score_selected_trie(forms, logits):
    """Drive the production selector and plan using declared scores, not a model."""
    prepared = {
        label: CandidatePaths(bare=bare, spaced=spaced) for label, (bare, spaced) in forms.items()
    }
    validate_potential_paths(prepared, end_tokens=END)
    wanted = first_token_ids(prepared)
    first = validate_readout(wanted, logits[()])
    selected = select_paths(prepared, first)
    plan = build_trie(selected, end_tokens=END)
    readouts = {(): first}
    for request in plan.additional_readouts:
        readouts[request.prefix] = logits[request.prefix]
    return selected, plan.score(readouts)


@given(case=selected_path_cases())
def test_selected_chain_rule_matches_independent_reference(case):
    forms, logits = case
    expected_paths, expected_logs = _naive_reference(forms, logits)
    paths, logs = _score_selected_trie(forms, logits)

    assert paths == expected_paths
    assert logs == pytest.approx(expected_logs, abs=1e-12)
    assert math.fsum(math.exp(value) for value in logs.values()) == pytest.approx(1.0, abs=1e-12)
    result = measurement_from_raw(RawReadout(logs, paths, 0))
    assert result.raw.credence_logprobs == logs
    assert math.isfinite(result.entropy)
    reversed_paths, reversed_logs = _score_selected_trie(
        dict(reversed(list(forms.items()))), logits
    )
    assert list(reversed_paths) == list(reversed(paths))
    assert reversed_paths == paths
    assert reversed_logs == pytest.approx(logs, abs=1e-12)


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
    forms = FORM_SETS[2]
    logits = {(): {token: -math.inf for token in range(10)}, (0,): {1: -math.inf, 5: -math.inf}}
    _, logs = _score_selected_trie(forms, logits)

    assert {label: math.exp(value) for label, value in logs.items()} == pytest.approx(
        {
            "very negative": 1 / 8,
            "negative": 1 / 4,
            "neutral": 1 / 4,
            "positive": 1 / 4,
            "very positive": 1 / 8,
        }
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


@pytest.mark.parametrize(
    ("first", "later", "expected_paths", "expected_credences"),
    [
        (
            {0: math.log(0.2), 4: math.log(0.4), 5: math.log(0.3)},
            {},
            {"positive": (4,), "position": (5,)},
            {"positive": 4 / 7, "position": 3 / 7},
        ),
        (
            {0: math.log(0.3), 4: math.log(0.4), 5: math.log(0.2)},
            {},
            {"positive": (4,), "position": (0, 2, 3)},
            {"positive": 4 / 7, "position": 3 / 7},
        ),
        (
            {0: math.log(0.4), 4: math.log(0.3), 5: math.log(0.2)},
            {(0,): {1: math.log(0.6), 2: math.log(0.4)}},
            {"positive": (0, 1), "position": (0, 2, 3)},
            {"positive": 0.6, "position": 0.4},
        ),
    ],
)
def test_positive_position_golden_shapes_only_score_selected_branches(
    first, later, expected_paths, expected_credences
):
    paths, logs = _score_selected_trie(FORM_SETS[1], {(): first, **later})

    assert paths == expected_paths
    assert {label: math.exp(value) for label, value in logs.items()} == pytest.approx(
        expected_credences
    )


@pytest.mark.parametrize("continuation", [-math.inf, -80.0, -40.0, -1000.0])
def test_grouped_end_roundoff_cannot_make_a_valid_measurement_fail(continuation):
    steps = masked_logprobs([1, 10, 11], {1: continuation, 10: -4.5, 11: -0.5})
    end_log = logprob_sum(steps[token] for token in END)
    result = measurement_from_raw(
        RawReadout({"A": end_log, "AB": steps[1]}, {"A": (0,), "AB": (0, 1)}, 0)
    )

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
