"""Turn model scores into probabilities, and represent measurement results.

This module has two parts:

- Helpers that turn a model's raw next-token scores into probabilities when
  only a few tokens are allowed: masked_logprobs, logprob_sum, logsumexp, and
  validate_readout.
- The result of a measurement: Measurement and RawReadout, the
  measurement_from_raw function that builds them, and two helpers for ties.

Nothing here runs a model; every function works on numbers you supply.

Terms used below:

- A *logit* is a model's raw, unnormalized score for a possible next token.
  Higher is more likely. ``-inf`` means the token is impossible. Pass logits
  as Python floats.
- A *log probability* is the natural log of a probability, so it is at most 0.
  Log probabilities stay precise for answers far too unlikely to represent as
  ordinary probabilities.
- A *credence* is the probability a measurement assigns to one candidate
  answer. A measurement's credences sum to 1 across its candidates.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from .errors import BackendReadoutError

# Shared tolerance for normalized event mass and complete result mass.
_MASS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RawReadout:
    """The exact numbers behind a Measurement, for auditing and storage.

    Most code only needs ``Measurement.credences``. Use this record when you
    need an answer's precise log credence, or want to know which token IDs
    were scored for each answer.

    Log credences matter for very unlikely answers. A credence such as
    e^-1000 rounds to 0.0 as an ordinary float, but its log credence stays
    -1000.0. Only an answer with exactly zero probability has a log credence
    of ``-inf``.

    Attributes:
        credence_logprobs: Each candidate's credence as a natural-log probability.
        scored_token_ids: The token IDs scored for each candidate, either the
            version written as-is or the version with a leading space. Prompt
            tokens are not included.
        context_token_count: Number of prompt tokens before the answer.

    Examples:
        Suppose a tokenizer encodes ``"very positive"`` as ``"very"`` (ID 10)
        followed by ``" positive"`` (ID 11), and ``"negative"`` as one token
        (ID 20). These IDs are made up for illustration.

        >>> raw = RawReadout(
        ...     credence_logprobs={"very positive": math.log(0.75), "negative": math.log(0.25)},
        ...     scored_token_ids={"very positive": (10, 11), "negative": (20,)},
        ...     context_token_count=7,
        ... )
        >>> raw.scored_token_ids["very positive"]
        (10, 11)
    """

    credence_logprobs: dict[str, float]
    scored_token_ids: dict[str, tuple[int, ...]]
    context_token_count: int


@dataclass(frozen=True)
class Measurement:
    """A model's credences over a set of candidate answers, with summaries.

    ``credences`` is the main result: a probability for each candidate,
    summing to 1. These probabilities are relative to the candidates you
    supplied. They say how the model divides its belief among those answers,
    not how likely it was to write something else, and they are not
    guaranteed to be calibrated.

    The three confidence summaries each run from 0 (no better than a uniform
    guess) to 1 (all probability on one candidate), but they measure different
    things. In the formulas, K is the number of candidates and p1 and p2 are
    the largest and second-largest credences.

    Build a measurement with measurement_from_raw. Its fields cannot be
    reassigned; treat the dictionaries inside as read-only too.

    Attributes:
        credences: Each candidate's probability, in candidate order.
        entropy: Shannon entropy of the credences, in bits. It is 0 when one
            candidate has all the probability and log2(K) when all are equal.
        top_labels: Every candidate tied for the highest credence, in candidate
            order. Usually one label. Ties are reported, never broken; see
            choose_top_label and top_label_weights.
        top_label_confidence: How far the top candidate is above a uniform
            guess: (p1 - 1/K) / (1 - 1/K).
        entropy_confidence: How concentrated the whole distribution is:
            1 - entropy / log2(K).
        margin_confidence: The top candidate's lead over the runner-up: p1 - p2.
        raw: Exact log credences and the token IDs scored for each candidate.
    """

    credences: dict[str, float]
    entropy: float
    top_labels: tuple[str, ...]
    top_label_confidence: float
    entropy_confidence: float
    margin_confidence: float
    raw: RawReadout

    def to_dict(self) -> dict[str, object]:
        """Convert the measurement to a dictionary you can save as JSON.

        The dictionary uses the same field names, with ``raw`` as a nested
        dictionary, and tuples become lists. JSON has no infinity, so a log
        credence of ``-inf`` (exactly zero probability) becomes ``None``.
        Finite log credences stay numbers, even when the ordinary credence
        has rounded to 0.0. Values are not rounded.

        The result is a new copy; editing it does not change the measurement.

        Returns:
            A dictionary that ``json.dumps(..., allow_nan=False)`` accepts.

        Raises:
            ValueError: A credence or summary is NaN or infinite, or a log
                credence is NaN or +inf. This happens only with a hand-built
                Measurement, not one from measurement_from_raw.

        Examples:
            A tiny but nonzero credence keeps its log value; an exactly zero
            credence becomes ``None``.

            >>> import json
            >>> raw = RawReadout(
            ...     {"A": 0.0, "tiny": -1000.0, "zero": -math.inf},
            ...     {"A": (1,), "tiny": (2,), "zero": (3,)},
            ...     7,
            ... )
            >>> result = measurement_from_raw(raw)
            >>> exported = result.to_dict()
            >>> exported["raw"]["credence_logprobs"]
            {'A': 0.0, 'tiny': -1000.0, 'zero': None}
            >>> exported["raw"]["scored_token_ids"]["A"]
            [1]
            >>> json.loads(json.dumps(exported, allow_nan=False))["top_labels"]
            ['A']
        """
        return {
            "credences": {
                label: _finite(value, "credences") for label, value in self.credences.items()
            },
            "entropy": _finite(self.entropy, "entropy"),
            "top_labels": list(self.top_labels),
            "top_label_confidence": _finite(self.top_label_confidence, "top_label_confidence"),
            "entropy_confidence": _finite(self.entropy_confidence, "entropy_confidence"),
            "margin_confidence": _finite(self.margin_confidence, "margin_confidence"),
            "raw": {
                "credence_logprobs": {
                    label: None if value == -math.inf else _finite(value, "credence_logprobs")
                    for label, value in self.raw.credence_logprobs.items()
                },
                "scored_token_ids": {
                    label: list(path) for label, path in self.raw.scored_token_ids.items()
                },
                "context_token_count": self.raw.context_token_count,
            },
        }


def validate_readout(wanted: Iterable[int], readout: Mapping[int, float]) -> dict[int, float]:
    """Check that a model returned a usable score for every requested token.

    Each requested token needs a logit that is a real number or ``-inf``. A
    missing score raises an error instead of being treated as zero
    probability; it usually means the model backend returned only its
    highest-scoring tokens. Scores for tokens you did not request are ignored.

    Args:
        wanted: Token IDs you requested. Repeated IDs are checked once.
        readout: The model's logits, keyed by token ID.

    Returns:
        Each requested token ID, once, with its score.

    Raises:
        BackendReadoutError: A requested score is missing, not a number, NaN,
            or +inf.

    Examples:
        >>> validate_readout([10, 10, 20], {10: 2.0, 20: -math.inf, 99: 8.0})
        {10: 2.0, 20: -inf}
    """
    scores = {}
    for token in dict.fromkeys(wanted):
        try:
            value = readout[token]
        except KeyError as error:
            raise BackendReadoutError(f"Missing requested score for token {token}") from error
        try:
            scores[token] = _log_weight(value, f"score for token {token}")
        except (TypeError, ValueError) as error:
            raise BackendReadoutError(str(error)) from error
    return scores


def masked_logprobs(allowed: Iterable[int], readout: Mapping[int, float]) -> dict[int, float]:
    """Compute next-token probabilities when only some tokens are allowed.

    This is a softmax over the allowed tokens alone: their logits are turned
    into probabilities that sum to 1, ignoring the rest of the vocabulary.
    Each step of a candidate answer is scored this way.

    Special cases:

    - If only one distinct token is allowed, it gets probability 1 and needs
      no score.
    - A token with logit ``-inf`` gets probability 0.
    - If every allowed token has logit ``-inf``, they share the probability
      equally.

    To score several stopping (END) tokens as one choice, include each of
    them in ``allowed`` and then add their results with logprob_sum.

    Args:
        allowed: Token IDs the model may choose from. Repeated IDs are counted once.
        readout: The model's logits, keyed by token ID. Can be empty when
            only one distinct token is allowed.

    Returns:
        The natural-log probability of each distinct allowed token.

    Raises:
        ValueError: ``allowed`` is empty.
        BackendReadoutError: A needed score is missing or invalid.

    Examples:
        Results are log probabilities; exponentiate them to see probabilities.

        >>> logs = masked_logprobs([10, 20], {10: 2.0, 20: 1.0})
        >>> {token: round(math.exp(value), 6) for token, value in logs.items()}
        {10: 0.731059, 20: 0.268941}
        >>> masked_logprobs([12], {})  # Forced: no score required.
        {12: 0.0}
        >>> fallback = masked_logprobs([10, 20], {10: -math.inf, 20: -math.inf})
        >>> [round(math.exp(value), 6) for value in fallback.values()]
        [0.5, 0.5]
    """
    tokens = tuple(dict.fromkeys(allowed))
    if not tokens:
        raise ValueError("The allowed token set must be nonempty")
    if len(tokens) == 1:
        return {tokens[0]: 0.0}
    scores = validate_readout(tokens, readout)
    values = tuple(scores.values())
    if all(value == -math.inf for value in values):
        return dict.fromkeys(tokens, -math.log(len(tokens)))
    maximum, correction = _shifted_normalizer(values)
    # Subtract the maximum first. score - (maximum + correction) would lose
    # the correction at large common offsets, even for perfectly equal scores.
    return {token: (score - maximum) - correction for token, score in scores.items()}


def logsumexp(values: Iterable[float]) -> float:
    """Add numbers that are stored as natural logs, and return the log of the sum.

    Computes log(exp(a) + exp(b) + ...) without overflowing on large inputs or
    losing small ones. The inputs are general weights, so the result can be
    positive. To add the probabilities of outcomes that make up one event,
    use logprob_sum, which also checks that the total is at most 1.

    Args:
        values: Natural-log weights, each finite or ``-inf``.

    Returns:
        The natural log of the total weight. Empty input, or input that is
        all ``-inf``, gives ``-inf``.

    Raises:
        TypeError: An input is not a real number.
        ValueError: An input is NaN, +inf, or too large to convert to a float.

    Examples:
        Two weights of 1 sum to 2, so the result is log(2).

        >>> round(logsumexp([0.0, 0.0]), 6)
        0.693147
        >>> logsumexp([])
        -inf
    """
    weights = tuple(_log_weight(value, "log weight") for value in values)
    if not weights or all(value == -math.inf for value in weights):
        return -math.inf
    maximum, correction = _shifted_normalizer(weights)
    return maximum + correction


def logprob_sum(values: Iterable[float]) -> float:
    """Combine the probabilities of several outcomes that count as one event.

    For example, if a model can stop an answer with either of two END tokens,
    the probability of stopping is the sum of both. Inputs and output are
    natural-log probabilities, not logits.

    The outcomes must not overlap, so their total is at most 1. A total just
    above 1 from floating-point roundoff (relative error up to 1e-12) is
    returned as 0.0, meaning probability 1. A larger total raises an error,
    because it means the inputs were not probabilities of separate outcomes.

    Args:
        values: Natural-log probabilities of non-overlapping outcomes, each at
            most 0.

    Returns:
        The natural-log probability of the combined event. Empty input, or
        input that is all ``-inf``, gives ``-inf``.

    Raises:
        TypeError: An input is not a real number.
        ValueError: An input is positive, NaN, or +inf, or the probabilities
            sum to more than 1.

    Examples:
        One token continues the answer and two END tokens stop it. Score all
        three together, then combine the two END tokens.

        >>> steps = masked_logprobs([11, 99, 100], {11: 0.0, 99: 0.0, 100: 0.0})
        >>> stopping = logprob_sum(steps[token] for token in (99, 100))
        >>> round(math.exp(stopping), 6)
        0.666667
    """
    probabilities = tuple(_log_weight(value, "log probability") for value in values)
    if any(value > 0 for value in probabilities):
        raise ValueError("A log probability cannot be positive")
    result = logsumexp(probabilities)
    if result > math.log1p(_MASS_TOLERANCE):
        raise ValueError("Grouped event probability exceeds 1")
    return min(0.0, result)


def measurement_from_raw(raw: RawReadout) -> Measurement:
    """Build a Measurement from each candidate's log credence.

    Use this when you have one natural-log probability per candidate, for
    example from TriePlan.score. It computes the credences, the winners, the
    entropy, and the three confidence summaries.

    The credences must already sum to 1, within a relative tolerance of
    1e-12. They are not rescaled for you: a total that is off means
    something upstream went wrong, and rescaling would hide it.

    Candidate order follows ``raw.credence_logprobs``. The measurement keeps
    its own copies of the input, so later changes to ``raw`` do not affect it.

    Args:
        raw: Log credences and scored token IDs for at least two candidates,
            with the same candidates in both mappings.

    Returns:
        The measurement.

    Raises:
        TypeError: A log credence is not a real number.
        ValueError: There are fewer than two candidates, the two mappings name
            different candidates, a log credence is positive, NaN, or +inf,
            or the credences do not sum to 1.

    Examples:
        >>> raw = RawReadout(
        ...     {"A": math.log(0.6), "B": math.log(0.2), "C": math.log(0.2)},
        ...     {"A": (10,), "B": (20,), "C": (30,)},
        ...     7,
        ... )
        >>> result = measurement_from_raw(raw)
        >>> result.top_labels
        ('A',)
        >>> round(result.top_label_confidence, 6)
        0.4
        >>> round(result.entropy_confidence, 6)
        0.135026
        >>> round(result.margin_confidence, 6)
        0.4
    """
    if len(raw.credence_logprobs) < 2:
        raise ValueError("A measurement requires at least two candidates")
    if raw.credence_logprobs.keys() != raw.scored_token_ids.keys():
        raise ValueError("Log credences and scored paths must have the same candidates")
    logs = {
        label: _log_weight(value, f"log credence for {label!r}")
        for label, value in raw.credence_logprobs.items()
    }
    if any(value > 0 for value in logs.values()):
        raise ValueError("A log credence cannot be positive")
    credences = {label: math.exp(value) for label, value in logs.items()}
    total = math.fsum(credences.values())
    if not math.isclose(total, 1.0, rel_tol=_MASS_TOLERANCE, abs_tol=0.0):
        raise ValueError(f"Credences must sum to 1; got {total!r}")
    maximum = max(logs.values())
    winners = tuple(label for label, value in logs.items() if value == maximum)
    entropy = -math.fsum(
        probability * logs[label] for label, probability in credences.items() if probability != 0.0
    ) / math.log(2)
    largest, second = sorted(credences.values(), reverse=True)[:2]
    count = len(credences)
    # frozen=True prevents field reassignment, not edits to dictionary contents.
    # Own the mappings so changes to the input cannot invalidate these summaries.
    snapshot = RawReadout(
        credence_logprobs=logs,
        scored_token_ids={label: tuple(raw.scored_token_ids[label]) for label in logs},
        context_token_count=raw.context_token_count,
    )
    return Measurement(
        credences=credences,
        entropy=entropy,
        top_labels=winners,
        top_label_confidence=_bounded_confidence((largest - 1 / count) / (1 - 1 / count)),
        entropy_confidence=_bounded_confidence(1 - entropy / math.log2(count)),
        margin_confidence=largest - second,
        raw=snapshot,
    )


class _ChoiceRng(Protocol):
    """Any random number generator with a ``choice`` method, such as random.Random."""

    def choice(self, seq: Sequence[str], /) -> str: ...


def choose_top_label(measurement: Measurement, *, rng: _ChoiceRng) -> str:
    """Pick a single winning label, breaking any tie at random.

    A measurement lists every tied winner in ``top_labels`` and never breaks
    ties itself. Use this when your application needs exactly one label.
    Pass a seeded ``random.Random`` to make the choice reproducible. The
    measurement is not changed.

    Args:
        measurement: The measurement to pick a winner from.
        rng: The random number generator to use, such as ``random.Random(0)``.

    Returns:
        One of the labels in ``measurement.top_labels``, chosen uniformly.

    Raises:
        ValueError: The measurement has no top labels.

    Examples:
        >>> import random
        >>> tied = measurement_from_raw(
        ...     RawReadout(
        ...         {"A": -math.log(2), "B": -math.log(2)},
        ...         {"A": (10,), "B": (20,)},
        ...         7,
        ...     )
        ... )
        >>> choose_top_label(tied, rng=random.Random(7)) in tied.top_labels
        True
    """
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return rng.choice(measurement.top_labels)


def top_label_weights(measurement: Measurement) -> dict[str, float]:
    """Split one win equally among tied winners, for counting results.

    Each of the N labels in ``top_labels`` gets weight 1/N; other labels are
    left out. Adding these dictionaries over many measurements counts how
    often each label won, with ties shared instead of broken at random.

    These weights are not credences. Two winners with credence 0.4 each get
    weight 0.5 each.

    Args:
        measurement: The measurement whose winners to count.

    Returns:
        Each winning label with its weight, in ``top_labels`` order. The
        weights sum to 1.

    Raises:
        ValueError: The measurement has no top labels.

    Examples:
        >>> result = measurement_from_raw(
        ...     RawReadout(
        ...         {"A": math.log(0.4), "B": math.log(0.4), "C": math.log(0.2)},
        ...         {"A": (10,), "B": (20,), "C": (30,)},
        ...         7,
        ...     )
        ... )
        >>> top_label_weights(result)
        {'A': 0.5, 'B': 0.5}
    """
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return dict.fromkeys(measurement.top_labels, 1 / len(measurement.top_labels))


def _shifted_normalizer(values: Sequence[float]) -> tuple[float, float]:
    """Return a finite maximum and the log normalizer relative to it."""
    # Subtracting the maximum keeps exponentials <= 1, preventing overflow.
    # Keep the correction separate: adding it to a large maximum can erase it.
    maximum = max(values)
    peak = values.index(maximum)
    # Exclude one maximum (weight 1), then use log1p to preserve corrections
    # smaller than machine epsilon instead of rounding 1 + tiny mass to 1.
    tail = math.fsum(
        math.exp(value - maximum) for index, value in enumerate(values) if index != peak
    )
    return maximum, math.log1p(tail)


def _log_weight(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number; got {value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} is outside the supported numeric range") from error
    if math.isnan(number) or number == math.inf:
        raise ValueError(f"{field} must be finite or -inf; got {number!r}")
    return number


def _bounded_confidence(value: float) -> float:
    return min(1.0, max(0.0, value))


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"{field} must contain finite values for JSON serialization; got {value!r}"
        )
    return value
