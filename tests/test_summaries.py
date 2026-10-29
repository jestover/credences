"""Confidence summaries must remain derived, ordered, and free of hidden randomness."""

import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credences import RawReadout, ScoredPath, choose_top_label, top_label_weights
from credences.probabilities import masked_logprobs, measurement_from_raw


def _raw(logs, **changes):
    """One scored path per candidate, carrying the candidate's whole credence."""
    fields = {
        "credence_logprobs": logs,
        "scored_paths": {
            label: (ScoredPath(("bare",), (index,), value),)
            for index, (label, value) in enumerate(logs.items())
        },
        "context_token_count": 7,
        "coverage": 1.0,
    }
    return RawReadout(**{**fields, **changes})


def _measurement(probabilities):
    return measurement_from_raw(
        _raw(
            {
                label: math.log(probability) if probability else -math.inf
                for label, probability in probabilities.items()
            }
        )
    )


@pytest.mark.parametrize("count", [2, 3, 5, 10, 100])
def test_uniform_vectors_have_zero_confidence_and_all_winners(count):
    labels = [str(index) for index in range(count)]
    result = _measurement(dict.fromkeys(labels, 1 / count))

    assert result.credences == pytest.approx(dict.fromkeys(labels, 1 / count))
    assert result.entropy == pytest.approx(math.log2(count))
    assert result.top_labels == tuple(labels)
    assert result.top_label_confidence == pytest.approx(0.0, abs=1e-14)
    assert result.entropy_confidence == pytest.approx(0.0, abs=1e-14)
    assert result.margin_confidence == 0.0
    assert 0 <= result.top_label_confidence <= 1
    assert 0 <= result.entropy_confidence <= 1


def test_point_mass_has_unit_confidence_and_zero_entropy_without_zero_times_infinity():
    result = _measurement({"A": 0.0, "B": 1.0, "C": 0.0})

    assert result.entropy == 0.0
    assert result.top_labels == ("B",)
    assert result.top_label_confidence == 1.0
    assert result.entropy_confidence == 1.0
    assert result.margin_confidence == 1.0


def test_binary_top_and_margin_equal_twice_inner_confidence():
    result = _measurement({"A": 0.8, "B": 0.2})

    assert result.top_label_confidence == pytest.approx(2 * abs(0.8 - 0.5))
    assert result.margin_confidence == pytest.approx(result.top_label_confidence)
    expected_entropy = -0.8 * math.log2(0.8) - 0.2 * math.log2(0.2)
    assert result.entropy == pytest.approx(expected_entropy)
    assert result.entropy_confidence == pytest.approx(1 - expected_entropy)


def test_multiclass_top_and_entropy_confidence_can_rank_vectors_differently():
    first = _measurement({"A": 0.60, "B": 0.20, "C": 0.20})
    second = _measurement({"A": 0.55, "B": 0.45, "C": 0.0})

    assert first.top_label_confidence > second.top_label_confidence
    assert first.entropy_confidence < second.entropy_confidence
    assert first.margin_confidence == pytest.approx(0.4)
    assert second.margin_confidence == pytest.approx(0.1)


@given(
    scores=st.lists(
        st.one_of(
            st.just(-math.inf),
            st.floats(
                min_value=-1000, max_value=1000, width=32, allow_nan=False, allow_infinity=False
            ),
        ),
        min_size=2,
        max_size=15,
    )
)
def test_summaries_match_definitions_and_are_invariant_to_candidate_order(scores):
    logs = masked_logprobs(list(range(len(scores))), dict(enumerate(scores)))
    raw = _raw({str(token): value for token, value in logs.items()})
    result = measurement_from_raw(raw)
    probabilities = list(result.credences.values())
    largest, second = sorted(probabilities, reverse=True)[:2]
    entropy = -math.fsum(p * math.log2(p) for p in probabilities if p)

    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert result.entropy == pytest.approx(entropy, abs=1e-12)
    assert result.top_label_confidence == pytest.approx(
        (largest - 1 / len(scores)) / (1 - 1 / len(scores)), abs=1e-12
    )
    assert result.entropy_confidence == pytest.approx(
        1 - entropy / math.log2(len(scores)), abs=1e-12
    )
    assert result.margin_confidence == largest - second
    assert result.top_labels == tuple(
        label for label, value in raw.credence_logprobs.items() if value == max(logs.values())
    )
    assert 0 <= result.top_label_confidence <= 1
    assert 0 <= result.entropy_confidence <= 1
    assert 0 <= result.margin_confidence <= 1

    reordered = measurement_from_raw(_raw(dict(reversed(list(raw.credence_logprobs.items())))))
    assert reordered.credences == result.credences
    assert reordered.entropy == result.entropy
    assert reordered.top_label_confidence == result.top_label_confidence
    assert reordered.entropy_confidence == result.entropy_confidence
    assert reordered.margin_confidence == result.margin_confidence
    assert reordered.top_labels == tuple(reversed(result.top_labels))


def test_exact_log_winners_do_not_become_probability_rounding_ties():
    # Adjacent log values can share a rounded probability, but the particular
    # pair differs between platform math libraries. Find a real local pair.
    first = math.log(0.4)
    for _ in range(64):
        second = math.nextafter(first, -math.inf)
        if math.exp(first) == math.exp(second):
            break
        first = second
    else:
        pytest.fail("Could not construct an adjacent-log probability rounding tie")
    result = measurement_from_raw(_raw({"A": first, "B": second, "C": math.log(0.2)}))

    assert result.credences["A"] == result.credences["B"]
    assert result.margin_confidence == 0.0
    assert result.top_labels == ("A",)


def test_factory_preserves_underflowed_finite_logs_and_exact_zeros_in_json():
    result = measurement_from_raw(_raw({"A": 0.0, "tiny": -1000.0, "zero": -math.inf}))
    exported = json.loads(json.dumps(result.to_dict(), allow_nan=False))

    assert result.credences == {"A": 1.0, "tiny": 0.0, "zero": 0.0}
    assert result.entropy == 0.0
    assert result.top_labels == ("A",)
    assert exported["raw"]["credence_logprobs"]["tiny"] == -1000.0
    assert exported["raw"]["credence_logprobs"]["zero"] is None


def test_factory_never_renormalizes_candidate_credences_or_rewrites_raw_logs():
    raw = _raw({"A": math.log(0.6), "B": math.log(0.4) + 5e-13})
    result = measurement_from_raw(raw)

    assert result.credences == {
        label: math.exp(value) for label, value in raw.credence_logprobs.items()
    }
    assert math.fsum(result.credences.values()) != 1.0
    assert result.raw == raw
    raw.credence_logprobs["A"] = -math.inf
    raw.scored_paths["A"] = (ScoredPath(("bare",), (99,), -math.inf),)
    assert result.raw.credence_logprobs["A"] == math.log(0.6)
    assert result.raw.scored_paths["A"][0].token_ids == (0,)


def test_factory_owns_scored_paths_built_from_caller_lists():
    spellings, tokens = ["bare"], [0]
    paths = {
        "A": (ScoredPath(spellings, tokens, math.log(0.6)),),
        "B": (ScoredPath(("bare",), (1,), math.log(0.4)),),
    }
    result = measurement_from_raw(
        _raw({"A": math.log(0.6), "B": math.log(0.4)}, scored_paths=paths)
    )
    spellings.append("spaced")
    tokens.append(99)

    assert result.raw.scored_paths["A"][0].spellings == ("bare",)
    assert result.raw.scored_paths["A"][0].token_ids == (0,)


def test_factory_rejects_mass_off_by_more_than_roundoff():
    # 5e-13 of excess log mass is roundoff (above); 1e-10 is a real error.
    with pytest.raises(ValueError, match="sum to"):
        measurement_from_raw(_raw({"A": math.log(0.6), "B": math.log(0.4) + 1e-10}))


def test_factory_sums_a_candidates_spellings_and_rejects_paths_that_disagree():
    logs = {"A": math.log(0.6), "B": math.log(0.4)}
    both = (
        ScoredPath(("bare",), (0,), math.log(0.45)),
        ScoredPath(("spaced",), (1,), math.log(0.15)),
    )
    paths = {"A": both, "B": (ScoredPath(("bare",), (2,), math.log(0.4)),)}
    assert measurement_from_raw(_raw(logs, scored_paths=paths)).credences["A"] == (
        pytest.approx(0.6)
    )

    paths["A"] = both[:1]
    with pytest.raises(ValueError, match="add up"):
        measurement_from_raw(_raw(logs, scored_paths=paths))


def test_factory_rejects_a_candidate_without_scored_paths():
    logs = {"A": math.log(0.6), "B": math.log(0.4)}
    paths = {"A": (), "B": (ScoredPath(("bare",), (2,), math.log(0.4)),)}

    with pytest.raises(ValueError, match="no scored path"):
        measurement_from_raw(_raw(logs, scored_paths=paths))


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"coverage": -0.1}, ValueError, "Coverage"),
        ({"coverage": 1.1}, ValueError, "Coverage"),
        ({"coverage": math.nan}, ValueError, "coverage"),
        ({"coverage": "high"}, TypeError, "coverage"),
        ({"context_token_count": -1}, ValueError, "token count"),
        ({"context_token_count": True}, ValueError, "token count"),
        ({"context_token_count": 1.5}, ValueError, "token count"),
    ],
)
def test_factory_rejects_invalid_coverage_and_prompt_token_counts(changes, error, match):
    with pytest.raises(error, match=match):
        measurement_from_raw(_raw({"A": math.log(0.6), "B": math.log(0.4)}, **changes))


@pytest.mark.parametrize("logs", [{}, {"A": 0.0}])
def test_factory_rejects_fewer_than_two_candidates(logs):
    with pytest.raises(ValueError, match="at least two"):
        measurement_from_raw(_raw(logs))


@pytest.mark.parametrize(
    "logs",
    [
        {"A": -math.inf, "B": -math.inf},
        {"A": math.log(0.8), "B": math.log(0.8)},
        {"A": math.log(0.2), "B": math.log(0.2)},
    ],
)
def test_factory_rejects_non_normalized_logs_instead_of_inventing_label_fallback(logs):
    with pytest.raises(ValueError, match="sum to"):
        measurement_from_raw(_raw(logs))


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.1])
def test_factory_rejects_invalid_or_positive_log_credences(value):
    with pytest.raises(ValueError, match="log credence"):
        measurement_from_raw(_raw({"A": value, "B": -math.inf}))


def test_factory_rejects_missing_candidate_paths():
    logs = {"A": -math.log(2), "B": -math.log(2)}
    paths = {"A": (ScoredPath(("bare",), (1,), -math.log(2)),)}

    with pytest.raises(ValueError, match="same candidates"):
        measurement_from_raw(_raw(logs, scored_paths=paths))


def test_fractional_winner_weights_are_not_credences():
    result = _measurement({"B": 0.4, "A": 0.4, "C": 0.2})
    weights = top_label_weights(result)

    assert weights == {"B": 0.5, "A": 0.5}
    assert list(weights) == ["B", "A"]
    assert math.fsum(weights.values()) == 1.0


def test_tie_selection_delegates_only_winners_to_the_callers_rng():
    result = _measurement({"B": 0.4, "A": 0.4, "C": 0.2})

    class RecordingRng:
        def choice(self, labels):
            assert labels == ("B", "A")
            return labels[-1]

    assert choose_top_label(result, rng=RecordingRng()) == "A"
