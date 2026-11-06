"""Score multi-token candidate answers using a model's next-token logits.

A language model predicts one token at a time, but a candidate answer can be
several tokens long, and different candidates can start with the same token.
Each candidate is also scored in two spellings, as written and after one
space, because whether a model writes that space depends on what comes before
the answer. This module works out which next-token scores you need from the
model, then combines them into one credence for each candidate: the total
probability of its two spellings.

You supply the token IDs for each spelling and get the logits from your model;
this module does no tokenization and never calls a model. The steps are:

1. Put each candidate's two encodings in a CandidatePaths.
2. Call build_trie. It checks the encodings and returns a TriePlan before any
   model call is made.
3. For each request in the plan's ``readouts``, get the model's TokenScores
   after the prompt followed by the request's ``prefix``.
4. Call the plan's score method to get a RawReadout, and measurement_from_raw
   to turn it into a Measurement.

Two terms come up throughout. A *token path* is the sequence of token IDs for
one spelling, counted from where scoring starts. *END tokens* are the token IDs
that mean an answer has finished, such as the model's end-of-turn token. They
are needed when one candidate is the start of another, as ``"up"`` is of
``"upward"``: after ``"up"``, the model chooses between stopping and continuing.

Logits should be Python floats, and ``-inf`` means the token is impossible.

Examples:
    Two one-token candidates, with made-up token IDs and logits:

    >>> forms = {
    ...     "yes": CandidatePaths(bare=(10,), spaced=(30,)),
    ...     "no": CandidatePaths(bare=(20,), spaced=(40,)),
    ... }
    >>> plan = build_trie(forms)
    >>> plan.readouts
    (ReadoutRequest(prefix=(), wanted=(10, 20, 30, 40)),)
    >>> scores = {(): TokenScores({10: 2.0, 20: 1.0, 30: 0.0, 40: 0.0}, log_normalizer=3.0)}
    >>> raw = plan.score(scores, context_token_count=7)
    >>> {label: round(math.exp(value), 3) for label, value in raw.credence_logprobs.items()}
    {'yes': 0.693, 'no': 0.307}
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import NewType

from .errors import BackendReadoutError
from .probabilities import _MASS_TOLERANCE, RawReadout, ScoredPath, TokenScores

type TokenPath = tuple[int, ...]

# A natural-log weight that is finite or -inf, never NaN or +inf. Only
# _log_weight makes one from a backend's number, so ty rejects unchecked floats
# passed to the scoring helpers at the end of this module.
_LogWeight = NewType("_LogWeight", float)

# Relative float32 roundoff allowed when a backend's full-vocabulary normalizer
# is compared with the requested logits.
_NORMALIZER_TOLERANCE = 1e-5


@dataclass(frozen=True, init=False)
class CandidatePaths:
    """The two token encodings of one candidate answer.

    Tokenizers usually encode a word differently with and without a space in
    front of it, and whether a model writes that space depends on what comes
    before the answer. So each candidate is given in both versions: ``bare``
    encodes the answer exactly as written, and ``spaced`` encodes it with one
    space in front. The two can differ in length and in every token, not just
    the first. Both are scored, and the candidate's credence is their total.

    Encode each version together with the prompt, the way the model would see
    the full text, and keep the tokens that follow the prompt. This class only
    stores the token IDs; it cannot check what they spell. If both versions give
    the same tokens, pass the same path twice; it is scored once.

    Attributes:
        bare: Token IDs for the answer exactly as written.
        spaced: Token IDs for the answer with one space in front.

    Examples:
        Suppose a tokenizer splits ``"positive"`` into three tokens but encodes
        ``" positive"`` as one token. The IDs are made up.

        >>> paths = CandidatePaths(bare=[973, 184, 1784], spaced=[17384])
        >>> paths.bare, paths.spaced
        ((973, 184, 1784), (17384,))
    """

    bare: TokenPath
    spaced: TokenPath

    def __init__(self, bare: Sequence[int], spaced: Sequence[int]) -> None:
        """Store both encodings as tuples.

        Args:
            bare: Token IDs for the answer exactly as written.
            spaced: Token IDs for the answer with one space in front.

        Raises:
            TypeError: An encoding is not a sequence of token IDs, for example a string.
            ValueError: An encoding is empty or contains an ID that is not a
                nonnegative integer.
        """
        # Copy into tuples so later edits to the caller's lists cannot change
        # this frozen object.
        object.__setattr__(self, "bare", _token_path(bare))
        object.__setattr__(self, "spaced", _token_path(spaced))


@dataclass(frozen=True)
class SpellingPath:
    """One distinct token path of a candidate and the spellings it stands for.

    Attributes:
        spellings: ``("bare",)``, ``("spaced",)``, or ``("bare", "spaced")`` when
            both spellings produce the same tokens.
        token_ids: The path's tokens, counted from where scoring starts.
    """

    spellings: tuple[str, ...]
    token_ids: TokenPath


@dataclass(frozen=True)
class ReadoutRequest:
    """One set of next-token scores to get from the model.

    Feed the model your prompt tokens followed by the tokens in ``prefix``,
    and read its logits for every token ID in ``wanted``, plus its
    full-vocabulary log normalizer. A score is needed for each requested token,
    not only for the model's highest-scoring tokens.

    Attributes:
        prefix: Answer tokens to add after the prompt. Empty means the scores
            right after the prompt.
        wanted: Token IDs whose logits are needed, including any END tokens.

    Examples:
        With prompt tokens ``(7, 8)``, this request asks for the logits of
        tokens 11, 99, and 100 after the input ``(7, 8, 10)``:

        >>> context = (7, 8)
        >>> request = ReadoutRequest(prefix=(10,), wanted=(11, 99, 100))
        >>> context + request.prefix, request.wanted
        ((7, 8, 10), (11, 99, 100))
    """

    prefix: TokenPath
    wanted: tuple[int, ...]


@dataclass(frozen=True)
class TriePlan:
    """The model scores needed to score a set of candidate answers.

    Create a plan with build_trie, get the scores it asks for, and pass them
    to score. The plan depends only on the token paths, never on the model's
    scores, so it can be built and inspected before any model call.

    It always asks for the scores right after the prompt, and after that only
    where paths diverge. Where only one next token is possible, that token has
    probability 1 and needs no model call; it still appears in the prefixes of
    later requests. A plan holds no prompt, scores, or model state, so the same
    plan can score any prompt that produces the same paths. Its fields cannot
    be reassigned; treat ``paths`` as read-only too.

    Attributes:
        paths: Each candidate label mapped to its distinct spelling paths, in
            candidate order, bare spelling first.
        end_tokens: Token IDs that mean an answer has finished, sorted, each once.
        readouts: Every request for scores, starting with the one right after
            the prompt, shortest prefix first.

    Examples:
        >>> plan = build_trie(
        ...     {"up": CandidatePaths((10,), (30,)), "down": CandidatePaths((20,), (40,))}
        ... )
        >>> [request.prefix for request in plan.readouts]
        [()]
    """

    paths: dict[str, tuple[SpellingPath, ...]]
    end_tokens: tuple[int, ...]
    readouts: tuple[ReadoutRequest, ...]

    def score(
        self, scores: Mapping[TokenPath, TokenScores], *, context_token_count: int
    ) -> RawReadout:
        """Combine the model's scores into each path's and candidate's log probability.

        Every score is checked before any is used. A path's probability is the
        product of the probabilities of each token along it. At each step, the
        probabilities are a softmax over only the tokens the candidates allow
        there. Where only one token is allowed, it has probability 1. If every
        allowed token has logit ``-inf``, they share the probability equally;
        otherwise a ``-inf`` token gets probability 0.

        Where one candidate ends and another continues, the shorter one's
        probability of stopping is the combined probability of all END tokens.
        A candidate's credence is the total of its paths, and the credences sum
        to 1 with no extra rescaling. Coverage comes from the scores right
        after the prompt.

        Args:
            scores: The model's TokenScores for each request, keyed by the
                request's ``prefix``; use ``()`` for the scores right after the
                prompt. Every requested logit is needed, even on paths that end
                up with zero probability; extra logits are ignored.
            context_token_count: Number of prompt tokens before scoring starts,
                recorded in the result.

        Returns:
            A RawReadout with each candidate's log credence and scored paths,
            in candidate order, and the coverage. ``-inf`` means exactly zero
            probability.

        Raises:
            BackendReadoutError: A requested prefix or logit is missing; a logit
                is not a number, or is NaN or +inf; or a log normalizer is not
                finite or is smaller than its requested tokens' combined weight.

        Examples:
            ``"up"`` and ``"upward"`` share token 10. After it, the model
            either continues with 11 or stops with END token 99 or 100. With
            equal logits there, stopping gets 2/3 of that probability because
            two END tokens can stop it. The spaced spellings, tokens 30 and 31,
            get no probability here.

            >>> forms = {
            ...     "up": CandidatePaths((10,), (30,)),
            ...     "upward": CandidatePaths((10, 11, 12), (31,)),
            ...     "down": CandidatePaths((20,), (40,)),
            ... }
            >>> plan = build_trie(forms, end_tokens=(99, 100))
            >>> root = {10: 2.0, 20: 2.0, 30: -math.inf, 31: -math.inf, 40: -math.inf}
            >>> scores = {
            ...     (): TokenScores(root, log_normalizer=3.0),
            ...     (10,): TokenScores({11: 0.0, 99: 0.0, 100: 0.0}, log_normalizer=2.0),
            ... }
            >>> raw = plan.score(scores, context_token_count=7)
            >>> {label: round(math.exp(value), 3) for label, value in raw.credence_logprobs.items()}
            {'up': 0.333, 'upward': 0.167, 'down': 0.5}
        """
        # The model's scores enter here, so check them all once. The arithmetic
        # below trusts them.
        checked = {}
        for request in self.readouts:
            try:
                readout = scores[request.prefix]
            except KeyError as error:
                raise BackendReadoutError(
                    f"Missing branch readout at prefix {request.prefix}"
                ) from error
            checked[request.prefix] = _validate_token_scores(request.wanted, readout)
        transitions = {prefix: _masked_logprobs(found.logits) for prefix, found in checked.items()}

        credence_logprobs = {}
        scored_paths = {}
        for label, paths in self.paths.items():
            logprobs = [self._path_logprob(path, transitions) for path in paths]
            scored_paths[label] = tuple(
                ScoredPath(path.spellings, path.token_ids, logprob)
                for path, logprob in zip(paths, logprobs, strict=True)
            )
            credence_logprobs[label] = _logprob_sum(logprobs)
        return RawReadout(
            credence_logprobs=credence_logprobs,
            scored_paths=scored_paths,
            context_token_count=context_token_count,
            coverage=checked[()].requested_probability,
        )

    def _path_logprob(
        self, path: SpellingPath, transitions: Mapping[TokenPath, Mapping[int, _LogWeight]]
    ) -> _LogWeight:
        """Add up a path's step log probabilities, as returned by _masked_logprobs."""
        tokens = path.token_ids
        # Prefixes without a readout have one allowed token, so they add log 1 = 0.
        factors = [
            transitions[tokens[:depth]][token]
            for depth, token in enumerate(tokens)
            if tokens[:depth] in transitions
        ]
        if tokens in transitions:
            # Another candidate continues here, so this path must choose to stop.
            factors.append(_logprob_sum(transitions[tokens][token] for token in self.end_tokens))
        return _LogWeight(math.fsum(factors))


def build_trie(forms: Mapping[str, CandidatePaths], *, end_tokens: Iterable[int] = ()) -> TriePlan:
    """Check candidate encodings and plan which model scores are needed.

    Run this before calling the model. Both spellings of every candidate go
    into one plan, so candidates (and spellings) that start with the same
    tokens share those steps: the probability of a shared start is counted
    once, then divided at the point where the paths diverge.

    It raises if:

    - there are fewer than two candidates, or a label is empty;
    - two candidates have an identical token path;
    - one spelling's path is the start of the same candidate's other path;
    - one candidate's path is the start of another candidate's path (as with
      ``"up"`` and ``"upward"``) and no END tokens are given, or an END token
      is also the token that continues the longer path. The model's choice to
      stop must use different tokens from its choice to continue.

    Args:
        forms: Each candidate label mapped to its CandidatePaths. At least two
            candidates are needed.
        end_tokens: Token IDs that mean an answer has finished. Needed only
            when one candidate's path can be the start of another's. Repeated
            IDs count once.

    Returns:
        A plan for these paths, keeping candidate order.

    Raises:
        TypeError: A candidate's encodings are not given as CandidatePaths.
        ValueError: One of the problems listed above, or an END token ID that
            is not a nonnegative integer.

    Examples:
        ``"up"`` is the start of ``"upward"``, so after token 10 the plan asks
        whether the model stops or continues. After token 11, token 12 is the
        only possibility, so it needs no score.

        >>> forms = {
        ...     "up": CandidatePaths((10,), (30,)),
        ...     "upward": CandidatePaths((10, 11, 12), (31,)),
        ...     "down": CandidatePaths((20,), (40,)),
        ... }
        >>> build_trie(forms, end_tokens=(99, 100)).readouts
        (ReadoutRequest(prefix=(), wanted=(10, 20, 30, 31, 40)), ReadoutRequest(prefix=(10,), wanted=(11, 99, 100)))
    """
    _validate_labels(forms)
    paths = {label: _spelling_paths(label, value) for label, value in forms.items()}
    owners = _path_owners(
        (label, path.token_ids) for label, spellings in paths.items() for path in spellings
    )
    ends = _end_tokens(end_tokens)
    # Map each prefix to the tokens that can follow it.
    children: dict[TokenPath, set[int]] = {}
    for tokens in owners:
        for depth, token in enumerate(tokens):
            children.setdefault(tokens[:depth], set()).add(token)

    readouts = []
    for prefix, allowed in children.items():
        if prefix in owners:
            # A candidate ends here while another continues, so stopping is a choice.
            _validate_stopping_boundary(prefix, allowed, ends)
            allowed.update(ends)
        # The root is always read: coverage is defined there.
        if len(allowed) > 1 or not prefix:
            readouts.append(ReadoutRequest(prefix, tuple(sorted(allowed))))
    readouts.sort(key=lambda request: (len(request.prefix), request.prefix))
    return TriePlan(paths=paths, end_tokens=ends, readouts=tuple(readouts))


def _spelling_paths(label: str, value: CandidatePaths) -> tuple[SpellingPath, ...]:
    if not isinstance(value, CandidatePaths):
        raise TypeError(f"Candidate {label!r} requires named bare and spaced paths")
    if value.bare == value.spaced:
        return (SpellingPath(("bare", "spaced"), value.bare),)
    for short, long in ((value.bare, value.spaced), (value.spaced, value.bare)):
        if long[: len(short)] == short:
            raise ValueError(
                f"Candidate {label!r} has one spelling path {short} that starts its other "
                f"spelling path {long}"
            )
    return (SpellingPath(("bare",), value.bare), SpellingPath(("spaced",), value.spaced))


def _path_owners(paths: Iterable[tuple[str, TokenPath]]) -> dict[TokenPath, str]:
    """Assign each path to one label, rejecting cross-label collisions."""
    owners = {}
    for label, path in paths:
        # A candidate's own paths are already distinct, so a repeat is another label's.
        if path in owners:
            raise ValueError(f"Path {path} is owned by both {owners[path]!r} and {label!r}")
        owners[path] = label
    return owners


def _validate_labels(paths: Mapping[str, object]) -> None:
    if len(paths) < 2:
        raise ValueError("At least two candidates are required")
    if any(not isinstance(label, str) or not label for label in paths):
        raise ValueError("Candidate labels must be nonempty strings")


def _token_path(values: Sequence[int]) -> TokenPath:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("A token path must be a sequence of integer token ids")
    path = _token_ids(values)
    if not path:
        raise ValueError("Token paths must be nonempty")
    return path


def _token_ids(values: Iterable[int]) -> tuple[int, ...]:
    tokens = tuple(values)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError("Token ids must be nonnegative integers")
    return tokens


def _end_tokens(values: Iterable[int]) -> tuple[int, ...]:
    """Return validated, sorted, distinct stopping-token IDs."""
    # Duplicate END IDs must not multiply stopping probability.
    return tuple(sorted(set(_token_ids(values))))


def _validate_stopping_boundary(
    prefix: TokenPath, continuations: Iterable[int], end_tokens: tuple[int, ...]
) -> None:
    """Require stopping tokens distinct from continuations at this prefix."""
    if not end_tokens:
        raise ValueError(f"Prefix {prefix} needs a nonempty END set")
    overlap = set(end_tokens).intersection(continuations)
    if overlap:
        raise ValueError(f"END overlaps continuation tokens {sorted(overlap)} at prefix {prefix}")


# Scoring arithmetic for TriePlan.score. Backend numbers enter only through
# _validate_token_scores, which turns each logit into a _LogWeight. The helpers
# after it take _LogWeight values and do not check them again.


@dataclass(frozen=True)
class _CheckedScores:
    """One backend readout that passed _validate_token_scores.

    Attributes:
        logits: Exactly the requested tokens' logits, in request order.
        requested_probability: Full-vocabulary probability of the requested
            tokens, from 0 to 1. Right after the prompt, this is the coverage.
    """

    logits: dict[int, _LogWeight]
    requested_probability: float


def _validate_token_scores(wanted: Iterable[int], scores: TokenScores) -> _CheckedScores:
    """Check one backend readout and keep the requested logits.

    Args:
        wanted: Token IDs that were requested.
        scores: What the backend returned for them.

    Returns:
        The requested logits and their full-vocabulary probability.

    Raises:
        BackendReadoutError: A requested logit is missing, not a real number,
            NaN, or +inf; or the log normalizer is not finite or is smaller
            than the requested tokens' combined weight.
    """
    logits = {}
    for token in wanted:
        try:
            value = scores.logits[token]
        except KeyError as error:
            raise BackendReadoutError(f"Missing requested score for token {token}") from error
        logits[token] = _log_weight(value, f"score for token {token}")
    normalizer = _log_weight(scores.log_normalizer, "log normalizer")
    if normalizer == -math.inf:
        raise BackendReadoutError("The log normalizer must be finite; got -inf")
    requested = _logsumexp(logits.values())
    # Backends compute the normalizer in float32, so allow its roundoff, and
    # never let it fall below the requested tokens' own weight.
    slack = _NORMALIZER_TOLERANCE * max(1.0, abs(normalizer))
    if requested > normalizer + slack:
        raise BackendReadoutError(
            f"The log normalizer {normalizer!r} is smaller than the requested tokens' "
            f"combined weight {requested!r}"
        )
    return _CheckedScores(logits, math.exp(requested - max(normalizer, requested)))


def _log_weight(value: float, name: str) -> _LogWeight:
    """Return a backend number as a _LogWeight.

    Raises:
        BackendReadoutError: ``value`` is not a real number, or is NaN or +inf.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BackendReadoutError(f"The {name} must be a real number; got {value!r}")
    number = float(value)
    if math.isnan(number) or number == math.inf:
        raise BackendReadoutError(f"The {name} must be finite or -inf; got {number!r}")
    return _LogWeight(number)


def _masked_logprobs(logits: Mapping[int, _LogWeight]) -> dict[int, _LogWeight]:
    """Return log probabilities from a softmax over only the given tokens.

    A single token gets probability 1 whatever its logit. If every logit is
    -inf, the tokens share the probability equally; otherwise a -inf token
    gets probability 0.

    Args:
        logits: Logits for at least one token; an empty mapping raises
            ValueError. NaN or +inf, which the _LogWeight type rules out, is
            not detected here: it would make the results NaN.

    Returns:
        Each token's natural-log probability, at most 0.
    """
    if len(logits) == 1:
        return dict.fromkeys(logits, _LogWeight(0.0))
    values = tuple(logits.values())
    if all(value == -math.inf for value in values):
        return dict.fromkeys(logits, _LogWeight(-math.log(len(values))))
    maximum, correction = _shifted_normalizer(values)
    # Subtract the maximum first. score - (maximum + correction) would lose
    # the correction at large common offsets, even for perfectly equal scores.
    return {token: _LogWeight((score - maximum) - correction) for token, score in logits.items()}


def _logsumexp(values: Iterable[_LogWeight]) -> _LogWeight:
    """Return log(exp(a) + exp(b) + ...) without overflow or losing small terms.

    The inputs are general weights, so the result can be positive.

    Args:
        values: Natural-log weights. Empty or all -inf input gives -inf. NaN
            or +inf, which the _LogWeight type rules out, is not detected
            here: it would give NaN or +inf.
    """
    weights = tuple(values)
    if all(value == -math.inf for value in weights):
        return _LogWeight(-math.inf)
    maximum, correction = _shifted_normalizer(weights)
    return _LogWeight(maximum + correction)


def _logprob_sum(values: Iterable[_LogWeight]) -> _LogWeight:
    """Return the log probability of an event made of non-overlapping outcomes.

    Args:
        values: Natural-log probabilities, each at most 0, of outcomes that
            do not overlap, so their total is at most 1. A total just above 1
            from roundoff is returned as 0, meaning probability 1. NaN, which
            the _LogWeight type rules out, would give NaN.

    Raises:
        ValueError: The total is clearly above 1: an outcome was counted twice
            or a value was positive.
    """
    result = _logsumexp(values)
    if result > math.log1p(_MASS_TOLERANCE):
        raise ValueError("Grouped event probability exceeds 1")
    # Keep result first: min(nan, 0.0) is nan, but min(0.0, nan) is 0.0, which
    # would turn a NaN into probability 1.
    return _LogWeight(min(result, 0.0))


def _shifted_normalizer(values: Sequence[_LogWeight]) -> tuple[float, float]:
    """Return the largest value and log(sum(exp(value - largest))).

    Args:
        values: Natural-log weights, at least one of them finite; callers
            handle the all -inf case first. Otherwise the result is NaN, or
            ValueError for no values.
    """
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
