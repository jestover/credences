"""Result persistence must not lose paths, exact ties, or tiny log credences."""

import json
import math
from dataclasses import FrozenInstanceError, replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credences import Measurement, RawReadout


@pytest.fixture
def tied_measurement() -> Measurement:
    return Measurement(
        credences={"positive": 0.5, "negative": 0.5},
        entropy=1.0,
        top_labels=("positive", "negative"),
        top_label_confidence=0.0,
        entropy_confidence=0.0,
        margin_confidence=0.0,
        raw=RawReadout(
            credence_logprobs={"positive": -math.log(2), "negative": -math.log(2)},
            scored_token_ids={"positive": (1, 2), "negative": (3,)},
            context_token_count=7,
        ),
    )


def test_json_schema_preserves_all_winners_and_selected_complete_paths(tied_measurement):
    exported = tied_measurement.to_dict()

    assert exported == {
        "credences": {"positive": 0.5, "negative": 0.5},
        "entropy": 1.0,
        "top_labels": ["positive", "negative"],
        "top_label_confidence": 0.0,
        "entropy_confidence": 0.0,
        "margin_confidence": 0.0,
        "raw": {
            "credence_logprobs": {"positive": -math.log(2), "negative": -math.log(2)},
            "scored_token_ids": {"positive": [1, 2], "negative": [3]},
            "context_token_count": 7,
        },
    }
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    assert list(exported["credences"]) == ["positive", "negative"]
    assert list(exported["raw"]["scored_token_ids"]) == ["positive", "negative"]


@given(logprob=st.floats(min_value=-1e6, max_value=-746, allow_infinity=False))
def test_serialization_distinguishes_underflow_from_exact_zero(logprob):
    measurement = Measurement(
        credences={"likely": 1.0, "tiny": math.exp(logprob), "impossible": 0.0},
        entropy=0.0,
        top_labels=("likely",),
        top_label_confidence=1.0,
        entropy_confidence=1.0,
        margin_confidence=1.0,
        raw=RawReadout(
            credence_logprobs={"likely": 0.0, "tiny": logprob, "impossible": -math.inf},
            scored_token_ids={"likely": (1,), "tiny": (2,), "impossible": (3,)},
            context_token_count=5,
        ),
    )

    restored = json.loads(json.dumps(measurement.to_dict(), allow_nan=False))

    assert restored["credences"]["tiny"] == 0.0
    assert restored["raw"]["credence_logprobs"]["tiny"] == logprob
    assert restored["raw"]["credence_logprobs"]["impossible"] is None
    assert measurement.raw.credence_logprobs["impossible"] == -math.inf


def test_exported_containers_cannot_mutate_the_measurement(tied_measurement):
    exported = tied_measurement.to_dict()
    exported["credences"]["positive"] = 0.0
    exported["top_labels"].clear()
    exported["raw"]["credence_logprobs"]["positive"] = None
    exported["raw"]["scored_token_ids"]["positive"].append(99)

    assert tied_measurement.credences["positive"] == 0.5
    assert tied_measurement.top_labels == ("positive", "negative")
    assert tied_measurement.raw.credence_logprobs["positive"] == -math.log(2)
    assert tied_measurement.raw.scored_token_ids["positive"] == (1, 2)
    assert tied_measurement.to_dict()["top_labels"] == ["positive", "negative"]


def test_result_fields_cannot_be_reassigned(tied_measurement):
    with pytest.raises(FrozenInstanceError):
        tied_measurement.entropy = 0.0
    with pytest.raises(FrozenInstanceError):
        tied_measurement.raw.context_token_count = 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    ["entropy", "top_label_confidence", "entropy_confidence", "margin_confidence"],
)
def test_nonfinite_summaries_cannot_escape_as_invalid_json(tied_measurement, field, value):
    malformed = replace(tied_measurement, **{field: value})

    with pytest.raises(ValueError, match=field):
        malformed.to_dict()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_credences_cannot_escape_as_invalid_json(tied_measurement, value):
    malformed = replace(tied_measurement, credences={"positive": value, "negative": 0.5})

    with pytest.raises(ValueError, match="credences"):
        malformed.to_dict()


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_only_negative_infinity_logprobs_can_become_null(tied_measurement, value):
    malformed_raw = replace(
        tied_measurement.raw,
        credence_logprobs={"positive": value, "negative": -math.log(2)},
    )

    with pytest.raises(ValueError, match="credence_logprobs"):
        replace(tied_measurement, raw=malformed_raw).to_dict()
