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

from .errors import BackendReadoutError
from .probabilities import (
    RawReadout,
    ScoredPath,
    TokenScores,
    coverage,
    logprob_sum,
    masked_logprobs,
    validate_token_scores,
)

type TokenPath = tuple[int, ...]


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
    plan can score any prompt that produces the same paths.

    Attributes:
        paths: Each candidate label with its distinct spelling paths, in
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

    paths: tuple[tuple[str, tuple[SpellingPath, ...]], ...]
    end_tokens: tuple[int, ...]
    readouts: tuple[ReadoutRequest, ...]

    def score(
        self, scores: Mapping[TokenPath, TokenScores], *, context_token_count: int
    ) -> RawReadout:
        """Combine the model's scores into each path's and candidate's log probability.

        A path's probability is the product of the probabilities of each token
        along it. At each step, the probabilities are computed among only the
        tokens the candidates allow there (see masked_logprobs). Where one
        candidate ends and another continues, the shorter one's probability of
        stopping is the combined probability of all END tokens. A candidate's
        credence is the total of its paths, and the credences sum to 1 with no
        extra rescaling. Coverage comes from the scores right after the prompt.

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
            BackendReadoutError: A requested prefix, logit, or normalizer is
                missing or invalid.

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
        transitions = {}
        for request in self.readouts:
            try:
                readout = scores[request.prefix]
            except KeyError as error:
                raise BackendReadoutError(
                    f"Missing branch readout at prefix {request.prefix}"
                ) from error
            validate_token_scores(request.wanted, readout)
            transitions[request.prefix] = masked_logprobs(request.wanted, readout.logits)

        credence_logprobs = {}
        scored_paths = {}
        for label, paths in self.paths:
            scored = tuple(
                ScoredPath(path.spellings, path.token_ids, self._path_logprob(path, transitions))
                for path in paths
            )
            scored_paths[label] = scored
            credence_logprobs[label] = logprob_sum(path.logprob for path in scored)
        root = self.readouts[0]
        return RawReadout(
            credence_logprobs=credence_logprobs,
            scored_paths=scored_paths,
            context_token_count=context_token_count,
            coverage=coverage(root.wanted, scores[root.prefix]),
        )

    def _path_logprob(
        self, path: SpellingPath, transitions: Mapping[TokenPath, Mapping[int, float]]
    ) -> float:
        tokens = path.token_ids
        # Prefixes without a readout have one allowed token, so they add log 1 = 0.
        factors = [
            transitions[tokens[:depth]][token]
            for depth, token in enumerate(tokens)
            if tokens[:depth] in transitions
        ]
        if tokens in transitions:
            # Another candidate continues here, so this path must choose to stop.
            factors.append(logprob_sum(transitions[tokens][token] for token in self.end_tokens))
        return math.fsum(factors)


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
    paths = tuple((label, _spelling_paths(label, value)) for label, value in forms.items())
    owners = _path_owners(
        (label, path.token_ids) for label, spellings in paths for path in spellings
    )
    ends = _end_tokens(end_tokens)
    children: dict[TokenPath, set[int]] = {}
    for tokens in owners:
        for depth, token in enumerate(tokens):
            children.setdefault(tokens[:depth], set()).add(token)

    readouts = []
    for prefix, next_tokens in children.items():
        allowed = set(next_tokens)
        if prefix in owners:
            _validate_stopping_boundary(prefix, next_tokens, ends)
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
        owner = owners.get(path)
        if owner is not None and owner != label:
            raise ValueError(f"Path {path} is owned by both {owner!r} and {label!r}")
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
