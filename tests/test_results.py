"""Result persistence must not lose paths, exact ties, or tiny log credences."""

import json
import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from credences import Measurement, RawReadout, ScoredPath


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
            scored_paths={
                "positive": (
                    ScoredPath(("bare",), (1, 2), math.log(0.4)),
                    ScoredPath(("spaced",), (4,), math.log(0.1)),
                ),
                "negative": (ScoredPath(("bare", "spaced"), (3,), -math.log(2)),),
            },
            context_token_count=7,
            coverage=0.25,
        ),
    )


def test_json_schema_preserves_all_winners_and_every_scored_path(tied_measurement):
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
            "scored_paths": {
                "positive": [
                    {"spellings": ["bare"], "token_ids": [1, 2], "logprob": math.log(0.4)},
                    {"spellings": ["spaced"], "token_ids": [4], "logprob": math.log(0.1)},
                ],
                "negative": [
                    {"spellings": ["bare", "spaced"], "token_ids": [3], "logprob": -math.log(2)}
                ],
            },
            "context_token_count": 7,
            "coverage": 0.25,
        },
    }
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    assert list(exported["credences"]) == ["positive", "negative"]
    assert list(exported["raw"]["scored_paths"]) == ["positive", "negative"]


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
            scored_paths={
                "likely": (ScoredPath(("bare",), (1,), 0.0),),
                "tiny": (ScoredPath(("bare",), (2,), logprob),),
                "impossible": (ScoredPath(("bare",), (3,), -math.inf),),
            },
            context_token_count=5,
            coverage=1.0,
        ),
    )

    restored = json.loads(json.dumps(measurement.to_dict(), allow_nan=False))

    assert restored["credences"]["tiny"] == 0.0
    assert restored["raw"]["credence_logprobs"]["tiny"] == logprob
    assert restored["raw"]["credence_logprobs"]["impossible"] is None
    assert restored["raw"]["scored_paths"]["tiny"][0]["logprob"] == logprob
    assert restored["raw"]["scored_paths"]["impossible"][0]["logprob"] is None
    assert measurement.raw.credence_logprobs["impossible"] == -math.inf


def test_exported_containers_cannot_mutate_the_measurement(tied_measurement):
    exported = tied_measurement.to_dict()
    exported["credences"]["positive"] = 0.0
    exported["top_labels"].clear()
    exported["raw"]["credence_logprobs"]["positive"] = None
    exported["raw"]["scored_paths"]["positive"][0]["token_ids"].append(99)
    exported["raw"]["scored_paths"]["positive"][0]["spellings"].clear()

    assert tied_measurement.credences["positive"] == 0.5
    assert tied_measurement.top_labels == ("positive", "negative")
    assert tied_measurement.raw.credence_logprobs["positive"] == -math.log(2)
    assert tied_measurement.raw.scored_paths["positive"][0].token_ids == (1, 2)
    assert tied_measurement.raw.scored_paths["positive"][0].spellings == ("bare",)
    assert tied_measurement.to_dict()["top_labels"] == ["positive", "negative"]


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


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_invalid_path_logprobs_cannot_escape_as_invalid_json(tied_measurement, value):
    paths = dict(tied_measurement.raw.scored_paths)
    paths["negative"] = (ScoredPath(("bare",), (3,), value),)
    malformed_raw = replace(tied_measurement.raw, scored_paths=paths)

    with pytest.raises(ValueError, match="scored_paths"):
        replace(tied_measurement, raw=malformed_raw).to_dict()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_coverage_cannot_escape_as_invalid_json(tied_measurement, value):
    malformed_raw = replace(tied_measurement.raw, coverage=value)

    with pytest.raises(ValueError, match="coverage"):
        replace(tied_measurement, raw=malformed_raw).to_dict()
