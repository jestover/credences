"""Turn model scores into probabilities, and represent measurement results.

This module has two parts:

- Helpers that turn a model's raw next-token scores into probabilities when
  only a few tokens are allowed: masked_logprobs, logprob_sum, logsumexp,
  validate_readout, and validate_token_scores. TokenScores is what a model
  backend reports, and coverage computes the diagnostic of the same name.
- The result of a measurement: Measurement, RawReadout and ScoredPath, the
  measurement_from_raw function that builds a Measurement, and two helpers
  for ties.

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
# Relative float32 roundoff allowed when a backend's full-vocabulary normalizer
# is compared with the requested logits.
_NORMALIZER_TOLERANCE = 1e-5


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
        Two requested tokens, in a vocabulary whose other tokens add up to
        exp(2.0) of unnormalized weight:

        >>> scores = TokenScores(
        ...     logits={10: 2.0, 20: 1.0},
        ...     log_normalizer=logsumexp([2.0, 1.0, 2.0]),
        ... )
        >>> round(coverage([10, 20], scores), 6)
        0.577681
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

        Raises:
            ValueError: A credence, summary, or coverage is NaN or infinite, or
                a log probability is NaN or +inf. This happens only with a
                hand-built Measurement, not one from measurement_from_raw.

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
                    label: _json_log(value, "credence_logprobs")
                    for label, value in self.raw.credence_logprobs.items()
                },
                "scored_paths": {
                    label: [
                        {
                            "spellings": list(path.spellings),
                            "token_ids": list(path.token_ids),
                            "logprob": _json_log(path.logprob, "scored_paths"),
                        }
                        for path in paths
                    ]
                    for label, paths in self.raw.scored_paths.items()
                },
                "context_token_count": self.raw.context_token_count,
                "coverage": _finite(self.raw.coverage, "coverage"),
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
    """Build a Measurement from scored paths and log credences.

    Usually ``raw`` comes from TriePlan.score. This computes the credences, the
    winners, the entropy, and the three confidence summaries.

    The input must be consistent: each candidate's log credence must equal the
    combined probability of its scored paths, and the credences must sum to 1,
    both within a relative tolerance of 1e-12. Nothing is rescaled for you: a
    total that is off means something upstream went wrong, and rescaling would
    hide it.

    Candidate order follows ``raw.credence_logprobs``. The measurement keeps
    its own copies of the input, so later changes to ``raw`` do not affect it.

    Args:
        raw: Log credences and scored paths for at least two candidates, with
            the same candidates in both mappings.

    Returns:
        The measurement.

    Raises:
        TypeError: A log probability or the coverage is not a real number.
        ValueError: There are fewer than two candidates; the two mappings name
            different candidates; a candidate has no scored path; a log
            probability is positive, NaN, or +inf; a candidate's paths do not
            add up to its credence; the credences do not sum to 1; coverage is
            outside 0 to 1; or the prompt token count is negative.

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
    if raw.credence_logprobs.keys() != raw.scored_paths.keys():
        raise ValueError("Log credences and scored paths must have the same candidates")
    logs = {
        label: _log_weight(value, f"log credence for {label!r}")
        for label, value in raw.credence_logprobs.items()
    }
    if any(value > 0 for value in logs.values()):
        raise ValueError("A log credence cannot be positive")
    paths = {label: _scored_paths(label, raw.scored_paths[label]) for label in logs}
    for label, value in logs.items():
        combined = logprob_sum(path.logprob for path in paths[label])
        if combined != value and not math.isclose(
            combined, value, rel_tol=_MASS_TOLERANCE, abs_tol=_MASS_TOLERANCE
        ):
            raise ValueError(
                f"Scored paths for {label!r} add up to log probability {combined!r}, "
                f"not its log credence {value!r}"
            )
    credences = {label: math.exp(value) for label, value in logs.items()}
    total = math.fsum(credences.values())
    if not math.isclose(total, 1.0, rel_tol=_MASS_TOLERANCE, abs_tol=0.0):
        raise ValueError(f"Credences must sum to 1; got {total!r}")
    coverage = _log_weight(raw.coverage, "coverage")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"Coverage must be between 0 and 1; got {coverage!r}")
    count_value = raw.context_token_count
    if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 0:
        raise ValueError(
            f"The prompt token count must be a nonnegative integer; got {count_value!r}"
        )
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
        scored_paths=paths,
        context_token_count=count_value,
        coverage=coverage,
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


def coverage(allowed: Iterable[int], scores: TokenScores) -> float:
    """Return the model's unrestricted probability of starting an allowed answer.

    This adds up the full-vocabulary probabilities of the ``allowed`` tokens at
    one position. It is a diagnostic about the model's habits, not about the
    accuracy of credences: a model that usually opens with "The" has low
    coverage, yet its credences can be exactly as good.

    Args:
        allowed: Token IDs that can begin an allowed answer.
        scores: The model's logits for those tokens and its full-vocabulary
            log normalizer at that position.

    Returns:
        A probability between 0 and 1.

    Raises:
        BackendReadoutError: A requested logit is missing or invalid, or the
            log normalizer is not finite or is smaller than the requested
            tokens' own combined weight.

    Examples:
        Allowed tokens with probabilities 0.07 and 0.02, when the model would
        rather write something else:

        >>> scores = TokenScores({1: math.log(0.07), 2: math.log(0.02)}, log_normalizer=0.0)
        >>> round(coverage([1, 2], scores), 6)
        0.09
    """
    logits = validate_readout(allowed, scores.logits)
    allowed_mass = logsumexp(logits.values())
    normalizer = _log_normalizer(scores.log_normalizer, allowed_mass)
    if allowed_mass == -math.inf:
        return 0.0
    return min(1.0, math.exp(allowed_mass - normalizer))


def validate_token_scores(wanted: Iterable[int], scores: TokenScores) -> dict[int, float]:
    """Check one backend readout: every requested logit and the normalizer.

    Args:
        wanted: Token IDs that were requested.
        scores: What the backend returned for them.

    Returns:
        Each requested token ID, once, with its logit.

    Raises:
        BackendReadoutError: A requested logit is missing or invalid, or the
            log normalizer is not finite or is smaller than the requested
            tokens' own combined weight.

    Examples:
        >>> validate_token_scores([10], TokenScores({10: 1.5}, log_normalizer=1.0))
        Traceback (most recent call last):
            ...
        credences.errors.BackendReadoutError: The log normalizer 1.0 is smaller than the requested tokens' combined weight 1.5
    """
    logits = validate_readout(wanted, scores.logits)
    _log_normalizer(scores.log_normalizer, logsumexp(logits.values()))
    return logits


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


def _scored_paths(label: str, paths: Sequence[ScoredPath]) -> tuple[ScoredPath, ...]:
    """Return validated, immutable copies of one candidate's scored paths."""
    if not paths:
        raise ValueError(f"Candidate {label!r} has no scored path")
    copies = []
    for path in paths:
        logprob = _log_weight(path.logprob, f"path log probability for {label!r}")
        if logprob > 0:
            raise ValueError(f"A path log probability for {label!r} cannot be positive")
        copies.append(ScoredPath(tuple(path.spellings), tuple(path.token_ids), logprob))
    return tuple(copies)


def _log_normalizer(value: float, requested_mass: float) -> float:
    """Validate a full-vocabulary log normalizer against the requested tokens' mass."""
    try:
        normalizer = _log_weight(value, "log normalizer")
    except (TypeError, ValueError) as error:
        raise BackendReadoutError(str(error)) from error
    if normalizer == -math.inf:
        raise BackendReadoutError("The log normalizer must be finite; got -inf")
    # Backends compute the normalizer in float32, so allow its roundoff.
    slack = _NORMALIZER_TOLERANCE * max(1.0, abs(normalizer))
    if requested_mass > normalizer + slack:
        raise BackendReadoutError(
            f"The log normalizer {normalizer!r} is smaller than the requested tokens' "
            f"combined weight {requested_mass!r}"
        )
    return max(normalizer, requested_mass)


def _json_log(value: float, field: str) -> float | None:
    """Return a log probability for JSON, with exact zero probability as None."""
    return None if value == -math.inf else _finite(value, field)


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"{field} must contain finite values for JSON serialization; got {value!r}"
        )
    return value
