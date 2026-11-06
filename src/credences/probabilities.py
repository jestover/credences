"""Represent the scores a model reports and the results of a measurement.

- TokenScores is what a model backend reports for one position.
- Measurement, RawReadout and ScoredPath hold the result of a measurement.
- measurement_from_raw builds a Measurement from a RawReadout, and
  choose_top_label and top_label_weights help when answers tie.

credences.trie turns a model's TokenScores into a RawReadout. Nothing here
runs a model; every function works on numbers you supply.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# Relative roundoff allowed in a total probability of 1: in a measurement's
# credences here, and when credences.trie combines outcomes into one event.
_MASS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TokenScores:
    """A model's scores for some next tokens, as a backend reports them.

    ``logits`` holds the requested tokens' raw logits. ``log_normalizer`` is the
    log of the sum of exp(logit) over the model's **whole** vocabulary at that
    position. Credences never use it; it only turns the requested logits into
    unrestricted probabilities for the coverage diagnostic.

    Attributes:
        logits: Raw logits keyed by token ID, one for every requested token.
        log_normalizer: Logsumexp of all the model's logits at this position.

    Examples:
        A token's unrestricted probability is exp(logit - log_normalizer):

        >>> scores = TokenScores(logits={10: 2.0, 20: 1.0}, log_normalizer=4.0)
        >>> round(math.exp(scores.logits[10] - scores.log_normalizer), 6)
        0.135335
    """

    logits: Mapping[int, float]
    log_normalizer: float


@dataclass(frozen=True)
class ScoredPath:
    """One scored way of writing a candidate, with its probability.

    A candidate is scored in two spellings: as written (``"bare"``) and after
    one space (``"spaced"``). Each spelling becomes a sequence of token IDs.
    When both spellings produce the same tokens, one path stands for both.

    Attributes:
        spellings: ``("bare",)``, ``("spaced",)``, or ``("bare", "spaced")``.
        token_ids: The path's tokens, counted from where scoring starts.
        logprob: The path's natural-log probability. A candidate's paths add
            up to its credence.

    Examples:
        >>> path = ScoredPath(("spaced",), (17384,), math.log(0.6))
        >>> round(math.exp(path.logprob), 3)
        0.6
    """

    spellings: tuple[str, ...]
    token_ids: tuple[int, ...]
    logprob: float


@dataclass(frozen=True)
class RawReadout:
    """The exact numbers behind a Measurement, for auditing and storage.

    Most code only needs ``Measurement.credences``. Use this record when you
    need an answer's precise log credence, want to see how much each spelling
    contributed, or want the coverage diagnostic.

    Log credences matter for very unlikely answers. A credence such as
    e^-1000 rounds to 0.0 as an ordinary float, but its log credence stays
    -1000.0. Only an answer with exactly zero probability has a log credence
    of ``-inf``.

    ``coverage`` is the model's unrestricted probability that its first answer
    token begins one of the allowed spellings. It describes the model's habits,
    not the measurement's accuracy: a model that likes to open with "The" can
    have low coverage and exactly the same credences as one that answers
    directly. Credences never depend on it.

    Attributes:
        credence_logprobs: Each candidate's credence as a natural-log probability.
        scored_paths: Each candidate's scored paths, bare spelling first. Their
            probabilities add up to the candidate's credence.
        context_token_count: Number of prompt tokens before scoring starts.
        coverage: Unrestricted probability, between 0 and 1, that the model's
            first answer token begins an allowed spelling.

    Examples:
        Suppose ``"yes"`` is token 10 and ``" yes"`` token 11, and ``"no"`` and
        ``" no"`` are tokens 20 and 21. These IDs are made up for illustration.

        >>> raw = RawReadout(
        ...     credence_logprobs={"yes": math.log(0.75), "no": math.log(0.25)},
        ...     scored_paths={
        ...         "yes": (
        ...             ScoredPath(("bare",), (10,), math.log(0.70)),
        ...             ScoredPath(("spaced",), (11,), math.log(0.05)),
        ...         ),
        ...         "no": (
        ...             ScoredPath(("bare",), (20,), math.log(0.24)),
        ...             ScoredPath(("spaced",), (21,), math.log(0.01)),
        ...         ),
        ...     },
        ...     context_token_count=7,
        ...     coverage=0.9,
        ... )
        >>> [path.token_ids for path in raw.scored_paths["yes"]]
        [(10,), (11,)]
    """

    credence_logprobs: dict[str, float]
    scored_paths: dict[str, tuple[ScoredPath, ...]]
    context_token_count: int
    coverage: float


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
        dictionary; each scored path becomes a dictionary, and tuples become
        lists. JSON has no infinity, so a log probability of ``-inf`` (exactly
        zero probability) becomes ``None``. Finite log probabilities stay
        numbers, even when the ordinary probability has rounded to 0.0. Values
        are not rounded.

        The result is a new copy; editing it does not change the measurement.

        Returns:
            A dictionary that ``json.dumps(..., allow_nan=False)`` accepts.

        Examples:
            A tiny but nonzero credence keeps its log value; an exactly zero
            credence becomes ``None``.

            >>> import json
            >>> def one_path(token, logprob):
            ...     return (ScoredPath(("bare",), (token,), logprob),)
            >>> raw = RawReadout(
            ...     credence_logprobs={"A": 0.0, "tiny": -1000.0, "zero": -math.inf},
            ...     scored_paths={
            ...         "A": one_path(1, 0.0),
            ...         "tiny": one_path(2, -1000.0),
            ...         "zero": one_path(3, -math.inf),
            ...     },
            ...     context_token_count=7,
            ...     coverage=0.5,
            ... )
            >>> exported = measurement_from_raw(raw).to_dict()
            >>> exported["raw"]["credence_logprobs"]
            {'A': 0.0, 'tiny': -1000.0, 'zero': None}
            >>> exported["raw"]["scored_paths"]["zero"]
            [{'spellings': ['bare'], 'token_ids': [3], 'logprob': None}]
            >>> json.loads(json.dumps(exported, allow_nan=False))["top_labels"]
            ['A']
        """
        return {
            "credences": dict(self.credences),
            "entropy": self.entropy,
            "top_labels": list(self.top_labels),
            "top_label_confidence": self.top_label_confidence,
            "entropy_confidence": self.entropy_confidence,
            "margin_confidence": self.margin_confidence,
            "raw": {
                "credence_logprobs": {
                    label: _json_log(value) for label, value in self.raw.credence_logprobs.items()
                },
                "scored_paths": {
                    label: [
                        {
                            "spellings": list(path.spellings),
                            "token_ids": list(path.token_ids),
                            "logprob": _json_log(path.logprob),
                        }
                        for path in paths
                    ]
                    for label, paths in self.raw.scored_paths.items()
                },
                "context_token_count": self.raw.context_token_count,
                "coverage": self.raw.coverage,
            },
        }


def measurement_from_raw(raw: RawReadout) -> Measurement:
    """Build a Measurement from scored paths and log credences.

    Usually ``raw`` comes from TriePlan.score. This computes the credences, the
    winners, the entropy, and the three confidence summaries.

    The credences must sum to 1 within a relative tolerance of 1e-12. Nothing
    is rescaled for you: a total that is off means something upstream went
    wrong, and rescaling would hide it.

    Candidate order follows ``raw.credence_logprobs``. The measurement keeps
    its own copies of the input's dictionaries, so later changes to ``raw`` do
    not affect it.

    Args:
        raw: Log credences and scored paths for at least two candidates.

    Returns:
        The measurement.

    Raises:
        ValueError: There are fewer than two candidates, or the credences do
            not sum to 1. Log credences that are NaN, +inf, or positive beyond
            that tolerance make the sum wrong, so they are rejected too.

    Examples:
        >>> def one_path(token, probability):
        ...     return (ScoredPath(("bare",), (token,), math.log(probability)),)
        >>> raw = RawReadout(
        ...     credence_logprobs={"A": math.log(0.6), "B": math.log(0.2), "C": math.log(0.2)},
        ...     scored_paths={
        ...         "A": one_path(10, 0.6),
        ...         "B": one_path(20, 0.2),
        ...         "C": one_path(30, 0.2),
        ...     },
        ...     context_token_count=7,
        ...     coverage=0.8,
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
    logs = dict(raw.credence_logprobs)
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
    # Copy the mappings so changes to the input cannot invalidate these summaries.
    snapshot = RawReadout(
        credence_logprobs=logs,
        scored_paths=dict(raw.scored_paths),
        context_token_count=raw.context_token_count,
        coverage=raw.coverage,
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
        >>> from credences.trie import CandidatePaths, build_trie
        >>> plan = build_trie({"A": CandidatePaths((1,), (2,)), "B": CandidatePaths((3,), (4,))})
        >>> logits = {1: 0.0, 2: -math.inf, 3: 0.0, 4: -math.inf}
        >>> tied = measurement_from_raw(
        ...     plan.score({(): TokenScores(logits, 1.0)}, context_token_count=7)
        ... )
        >>> tied.top_labels
        ('A', 'B')
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
        >>> from credences.trie import CandidatePaths, build_trie
        >>> forms = {label: CandidatePaths((i,), (i + 10,)) for i, label in enumerate("ABC")}
        >>> logits = {0: math.log(0.4), 1: math.log(0.4), 2: math.log(0.2)}
        >>> logits |= dict.fromkeys((10, 11, 12), -math.inf)
        >>> raw = build_trie(forms).score({(): TokenScores(logits, 0.0)}, context_token_count=7)
        >>> top_label_weights(measurement_from_raw(raw))
        {'A': 0.5, 'B': 0.5}
    """
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return dict.fromkeys(measurement.top_labels, 1 / len(measurement.top_labels))


def _bounded_confidence(value: float) -> float:
    return min(1.0, max(0.0, value))


def _json_log(value: float) -> float | None:
    """Return a log probability for JSON, with exact zero probability as None."""
    return None if value == -math.inf else value
