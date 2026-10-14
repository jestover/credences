"""Score multi-token candidate answers using a model's next-token logits.

A language model predicts one token at a time, but a candidate answer can be
several tokens long, and different candidates can start with the same token.
This module works out which next-token scores you need from the model, then
combines them into one credence for each candidate.

You supply the token IDs for each candidate and get the logits from your
model; this module does no tokenization and never calls a model. The steps
are:

1. Put each candidate's two encodings (with and without a leading space) in
   a CandidatePaths, and call validate_potential_paths to catch problems
   before running the model.
2. Get the model's logits for first_token_ids(forms), right after the prompt.
3. Call select_paths to choose which encoding to score for each candidate.
4. Call build_trie on the chosen encodings. For each request in the returned
   plan's ``additional_readouts``, get logits from the model.
5. Call the plan's score method to get each candidate's natural-log credence.

Two terms come up throughout. A *token path* is the sequence of token IDs for
one answer, as it would appear right after the prompt. *END tokens* are the
token IDs that mean an answer has finished, such as the model's end-of-turn
token. They are needed when one candidate is the start of another, as
``"up"`` is of ``"upward"``: after ``"up"``, the model chooses between
stopping and continuing.

Logits should be Python floats, and ``-inf`` means the token is impossible.

Examples:
    Two one-token candidates, with made-up token IDs and logits:

    >>> forms = {
    ...     "yes": CandidatePaths((10,), (30,)),
    ...     "no": CandidatePaths((20,), (40,)),
    ... }
    >>> validate_potential_paths(forms)
    >>> first_token_ids(forms)
    (10, 20, 30, 40)
    >>> initial = {10: 2.0, 30: 1.0, 20: 2.0, 40: 1.0}
    >>> plan = build_trie(select_paths(forms, initial))
    >>> logs = plan.score({(): initial})
    >>> {label: round(math.exp(value), 3) for label, value in logs.items()}
    {'yes': 0.5, 'no': 0.5}
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .errors import BackendReadoutError
from .probabilities import logprob_sum, masked_logprobs, validate_readout

type TokenPath = tuple[int, ...]


@dataclass(frozen=True, init=False)
class CandidatePaths:
    """The two token encodings of one candidate answer.

    Tokenizers usually encode a word differently with and without a space in
    front of it, and right after a prompt a model may prefer either one. So
    each candidate is given in both versions: ``bare`` encodes the answer
    exactly as written, and ``spaced`` encodes it with one space in front.
    The two can differ in length and in every token, not just the first.
    select_paths later picks which version to score.

    Encode each version as it would appear right after the prompt, and check
    that it decodes back to the intended text. This class only stores the
    token IDs; it cannot check what they spell.

    Attributes:
        bare: Token IDs for the answer exactly as written.
        spaced: Token IDs for the answer with one space in front.

    Examples:
        Suppose a tokenizer splits ``"positive"`` into three tokens but encodes
        ``" positive"`` as one token. The IDs and logits are made up.

        >>> forms = {
        ...     "positive": CandidatePaths((973, 184, 207), (17384,)),
        ...     "negative": CandidatePaths((42, 43), (84,)),
        ... }
        >>> select_paths(forms, {973: 1.0, 17384: 2.0, 42: 2.0, 84: 1.0})
        {'positive': (17384,), 'negative': (42, 43)}
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
class ReadoutRequest:
    """One set of next-token scores to get from the model.

    Feed the model your prompt tokens followed by the tokens in ``prefix``,
    and read its logits for every token ID in ``wanted``. A score is needed
    for each one, not only for the model's highest-scoring tokens.

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
    """The model scores needed to tell a set of candidate answers apart.

    Create a plan with build_trie, get the scores it asks for, and pass them
    to score to get credences.

    The plan only asks for scores where the candidates' token paths diverge.
    Where only one next token is possible, that token has probability 1 and
    needs no model call; it still appears in the prefixes of later requests.
    A plan holds no prompt, scores, or model state, so the same plan can
    score any prompt for which select_paths chose the same paths.

    Attributes:
        paths: Each candidate label with the token path being scored, in
            candidate order.
        end_tokens: Token IDs that mean an answer has finished, sorted, each once.
        readouts: Every request for scores, shortest prefix first.

    Examples:
        >>> plan = build_trie({"up": (10,), "down": (20,)})
        >>> logs = plan.score({(): {10: 2.0, 20: 2.0}})
        >>> {label: round(math.exp(value), 3) for label, value in logs.items()}
        {'up': 0.5, 'down': 0.5}
    """

    paths: tuple[tuple[str, TokenPath], ...]
    end_tokens: tuple[int, ...]
    readouts: tuple[ReadoutRequest, ...]

    @property
    def additional_readouts(self) -> tuple[ReadoutRequest, ...]:
        """The requests still to send to the model after the first-token scores.

        This is ``readouts`` without the request at the empty prefix. You
        already have the scores right after the prompt from choosing paths, so
        pass those to score under the key ``()`` instead of asking again.

        Returns:
            The remaining requests, shortest prefix first.

        Examples:
            >>> plan = build_trie(
            ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
            ...     end_tokens=(99, 100),
            ... )
            >>> context = (7, 8)
            >>> (request,) = plan.additional_readouts
            >>> context + request.prefix, request.wanted
            ((7, 8, 10), (11, 99, 100))
        """
        return tuple(request for request in self.readouts if request.prefix)

    def score(self, readouts: Mapping[TokenPath, Mapping[int, float]]) -> dict[str, float]:
        """Combine the model's scores into a log credence for each candidate.

        A candidate's credence is the product of the probabilities of each
        token along its path. At each step, the probabilities are computed
        among only the tokens the candidates allow there (see
        masked_logprobs). Where one candidate ends and another continues, the
        shorter one's probability of stopping is the combined probability of
        all END tokens. The credences sum to 1 with no extra rescaling.

        Args:
            readouts: The model's logits for each request, keyed first by the
                request's ``prefix`` and then by token ID. Use ``()`` for the
                scores right after the prompt. Every requested score is needed,
                even on paths that end up with zero probability; extra scores
                are ignored.

        Returns:
            Each candidate's natural-log credence, in candidate order. ``-inf``
            means exactly zero probability.

        Raises:
            BackendReadoutError: A requested prefix or score is missing, or a
                score is invalid.

        Examples:
            ``"up"`` and ``"upward"`` share token 10. After it, the model
            either continues with 11 or stops with END token 99 or 100. With
            equal logits, stopping gets 2/3 of that probability because two
            END tokens can stop it:

            >>> plan = build_trie(
            ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
            ...     end_tokens=(99, 100),
            ... )
            >>> logs = plan.score(
            ...     {
            ...         (): {10: 2.0, 20: 2.0},
            ...         (10,): {11: 0.0, 99: 0.0, 100: 0.0},
            ...     }
            ... )
            >>> {label: round(math.exp(value), 3) for label, value in logs.items()}
            {'up': 0.333, 'upward': 0.167, 'down': 0.5}
        """
        transitions = {}
        for request in self.readouts:
            try:
                scores = readouts[request.prefix]
            except KeyError as error:
                raise BackendReadoutError(
                    f"Missing branch readout at prefix {request.prefix}"
                ) from error
            transitions[request.prefix] = masked_logprobs(request.wanted, scores)

        result = {}
        for label, path in self.paths:
            factors = []
            for request in self.readouts:
                prefix = request.prefix
                if path[: len(prefix)] != prefix:
                    continue
                if len(path) > len(prefix):
                    factors.append(transitions[prefix][path[len(prefix)]])
                elif path == prefix:
                    factors.append(
                        logprob_sum(transitions[prefix][token] for token in self.end_tokens)
                    )
            result[label] = math.fsum(factors)
        return result


def first_token_ids(forms: Mapping[str, CandidatePaths]) -> tuple[int, ...]:
    """Return the token IDs whose scores you need right after the prompt.

    These are the first tokens of both encodings of every candidate, each
    listed once. Get the model's logits for them and pass those to
    select_paths. Call validate_potential_paths first: this function does not
    check END tokens.

    Args:
        forms: Each candidate label mapped to its CandidatePaths. At least two
            candidates are needed.

    Returns:
        The distinct first-token IDs, sorted.

    Raises:
        TypeError: A candidate's encodings are not given as CandidatePaths.
        ValueError: There are fewer than two candidates, a label is empty, or
            two candidates share an identical token path.

    Examples:
        >>> forms = {
        ...     "up": CandidatePaths((10,), (30,)),
        ...     "upward": CandidatePaths((10, 11, 12), (31,)),
        ...     "down": CandidatePaths((20,), (40,)),
        ... }
        >>> first_token_ids(forms)  # Shared start 10 is requested only once.
        (10, 20, 30, 31, 40)
    """
    owners = _path_owners(_possible_paths(forms))
    return tuple(sorted({path[0] for path in owners}))


def select_paths(
    forms: Mapping[str, CandidatePaths], first_scores: Mapping[int, float]
) -> dict[str, TokenPath]:
    """Choose which encoding, bare or spaced, to score for each candidate.

    Each candidate keeps the encoding whose first token has the higher logit
    right after the prompt; on an exact tie, it keeps the bare one. The
    chosen encoding is kept whole, whatever its length. The choice only
    decides how each answer is represented. It does not change that answer's
    probability.

    Args:
        forms: Each candidate label mapped to its CandidatePaths. At least two
            candidates are needed.
        first_scores: The model's logits right after the prompt, for every ID
            from first_token_ids, including tokens of encodings that end up
            not chosen.

    Returns:
        Each candidate label mapped to its chosen token path, in candidate order.

    Raises:
        TypeError: A candidate's encodings are not given as CandidatePaths.
        ValueError: There are fewer than two candidates, a label is empty, or
            two candidates share an identical token path.
        BackendReadoutError: A needed first-token score is missing or invalid.

    Examples:
        ``"upward"`` has a three-token bare encoding and a one-token spaced
        one. The spaced encoding wins only when its first token scores higher.

        >>> forms = {
        ...     "upward": CandidatePaths((10, 11, 12), (31,)),
        ...     "down": CandidatePaths((20,), (40,)),
        ... }
        >>> select_paths(forms, {10: 2.0, 31: 2.0, 20: 2.0, 40: 1.0})
        {'upward': (10, 11, 12), 'down': (20,)}
        >>> select_paths(forms, {10: 2.0, 31: 3.0, 20: 2.0, 40: 1.0})
        {'upward': (31,), 'down': (20,)}
    """
    scores = validate_readout(first_token_ids(forms), first_scores)
    return {
        label: paths.spaced if scores[paths.spaced[0]] > scores[paths.bare[0]] else paths.bare
        for label, paths in forms.items()
    }


def validate_potential_paths(
    forms: Mapping[str, CandidatePaths], *, end_tokens: Iterable[int] = ()
) -> None:
    """Check the candidates' encodings for problems before you call the model.

    Either encoding of a candidate might be chosen later, so this checks
    both. It raises if:

    - there are fewer than two candidates, or a label is empty;
    - two candidates have an identical token path;
    - one candidate's path is the start of another candidate's path (as with
      ``"up"`` and ``"upward"``) and no END tokens are given, or an END token
      is also the token that continues the longer path. The model's choice to
      stop must use different tokens from its choice to continue, or the two
      cannot be told apart.

    The two encodings of the same candidate may overlap freely, since only
    one of them is scored. This does not check what the token IDs spell.

    Args:
        forms: Each candidate label mapped to its CandidatePaths.
        end_tokens: Token IDs that mean an answer has finished. Needed only
            when one candidate's path can be the start of another's.

    Raises:
        TypeError: A candidate's encodings are not given as CandidatePaths.
        ValueError: One of the problems listed above, or an END token ID that
            is not a nonnegative integer.

    Examples:
        >>> forms = {
        ...     "up": CandidatePaths((10,), (30,)),
        ...     "upward": CandidatePaths((10, 11, 12), (31,)),
        ... }
        >>> validate_potential_paths(forms)
        Traceback (most recent call last):
            ...
        ValueError: Prefix (10,) needs a nonempty END set
        >>> validate_potential_paths(forms, end_tokens=(99, 100))
    """
    ends = _end_tokens(end_tokens)
    owners = _path_owners(_possible_paths(forms))
    for path, label in owners.items():
        for depth in range(1, len(path)):
            prefix = path[:depth]
            shorter_label = owners.get(prefix)
            if shorter_label is not None and shorter_label != label:
                _validate_stopping_boundary(prefix, (path[depth],), ends)


def build_trie(paths: Mapping[str, Sequence[int]], *, end_tokens: Iterable[int] = ()) -> TriePlan:
    """Plan which model scores are needed to score the chosen token paths.

    Usually ``paths`` is the result of select_paths. Candidates that start
    with the same tokens share those steps, so the probability of a shared
    start is counted once rather than once per candidate.

    Args:
        paths: At least two candidate labels, each mapped to a nonempty token path.
        end_tokens: Token IDs that mean an answer has finished. Needed when one
            path is the start of another, and must not include the token that
            continues the longer path at that point. Repeated IDs count once.

    Returns:
        A plan for these paths, keeping candidate order.

    Raises:
        TypeError: A path is not a sequence of token IDs.
        ValueError: There are fewer than two candidates, a label or path is
            empty, a token ID is not a nonnegative integer, two candidates
            have identical paths, or END tokens are missing or overlap a
            continuing token where they are needed.

    Examples:
        ``"up"`` is the start of ``"upward"``, so after token 10 the plan asks
        whether the model stops or continues. After token 11, token 12 is the
        only possibility, so it needs no score.

        >>> plan = build_trie(
        ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
        ...     end_tokens=(99, 100),
        ... )
        >>> plan.additional_readouts
        (ReadoutRequest(prefix=(10,), wanted=(11, 99, 100)),)
    """
    _validate_labels(paths)
    selected = tuple((label, _token_path(path)) for label, path in paths.items())
    owners = _path_owners(selected)
    ends = _end_tokens(end_tokens)
    children: dict[TokenPath, set[int]] = {}
    for _, path in selected:
        for depth, token in enumerate(path):
            children.setdefault(path[:depth], set()).add(token)

    readouts = []
    for prefix, next_tokens in children.items():
        allowed = set(next_tokens)
        if prefix in owners:
            _validate_stopping_boundary(prefix, next_tokens, ends)
            allowed.update(ends)
        if len(allowed) > 1:
            readouts.append(ReadoutRequest(prefix, tuple(sorted(allowed))))
    readouts.sort(key=lambda request: (len(request.prefix), request.prefix))
    return TriePlan(paths=selected, end_tokens=ends, readouts=tuple(readouts))


def _possible_paths(forms: Mapping[str, CandidatePaths]) -> tuple[tuple[str, TokenPath], ...]:
    _validate_labels(forms)
    result = []
    for label, paths in forms.items():
        if not isinstance(paths, CandidatePaths):
            raise TypeError(f"Candidate {label!r} requires named bare and spaced paths")
        result.extend(((label, paths.bare), (label, paths.spaced)))
    return tuple(result)


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
