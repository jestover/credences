"""Path ownership, stopping boundaries, summed spellings, and exact readout requirements."""

import math

import pytest
from fakes import score_with_backend

from credences.errors import BackendReadoutError
from credences.probabilities import TokenScores, measurement_from_raw
from credences.trie import CandidatePaths, ReadoutRequest, SpellingPath, _logsumexp, build_trie

CONTEXT = (90, 91)


class DeclaredScores:
    """Record requests and return declared logits at exact token contexts."""

    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def next_token_logits(self, token_ids, wanted):
        context = tuple(token_ids)
        self.calls.append((context, tuple(wanted)))
        assert context in self.tables, f"Unexpected readout at {context}"
        table = self.tables[context]
        return TokenScores(
            logits={token: table[token] for token in wanted if token in table},
            # One unlisted vocabulary token with logit 0 keeps the normalizer finite.
            log_normalizer=_logsumexp([*table.values(), 0.0]),
        )


def _forms(**spellings):
    return {label: CandidatePaths(bare, spaced) for label, (bare, spaced) in spellings.items()}


def _probabilities(raw):
    return {label: math.exp(value) for label, value in raw.credence_logprobs.items()}


def test_both_spellings_add_up_in_the_five_candidate_golden_example():
    forms = {
        "very negative": CandidatePaths((0, 1), (6, 1)),
        "negative": CandidatePaths((2,), (7,)),
        "neutral": CandidatePaths((3,), (8,)),
        "positive": CandidatePaths((4,), (9,)),
        "very positive": CandidatePaths((0, 5), (6, 5)),
    }
    root = {0: 0.10, 6: 0.001, 2: 0.65, 7: 0.002, 3: 0.20, 8: 0.001, 4: 0.02, 9: 0.001}
    backend = DeclaredScores(
        {
            CONTEXT: {token: math.log(p) for token, p in root.items()},
            (*CONTEXT, 0): {1: math.log(0.95), 5: math.log(0.01)},
            (*CONTEXT, 6): {1: math.log(0.90), 5: math.log(0.05)},
        }
    )
    _, raw = score_with_backend(backend, CONTEXT, forms)

    expected = {
        "very negative": (0.10 / 0.975) * (0.95 / 0.96) + (0.001 / 0.975) * (0.90 / 0.95),
        "negative": 0.652 / 0.975,
        "neutral": 0.201 / 0.975,
        "positive": 0.021 / 0.975,
        "very positive": (0.10 / 0.975) * (0.01 / 0.96) + (0.001 / 0.975) * (0.05 / 0.95),
    }
    assert _probabilities(raw) == pytest.approx(expected)
    # Keeping only the stronger spelling would give 0.65 / 0.97 = 0.6701 instead.
    assert _probabilities(raw)["negative"] == pytest.approx(0.6687, abs=1e-4)
    assert [path.spellings for path in raw.scored_paths["very negative"]] == [
        ("bare",),
        ("spaced",),
    ]
    assert backend.calls == [
        (CONTEXT, (0, 2, 3, 4, 6, 7, 8, 9)),
        ((*CONTEXT, 0), (1, 5)),
        ((*CONTEXT, 6), (1, 5)),
    ]


def test_identical_spellings_are_one_path_counted_once():
    forms = _forms(A=((0,), (0,)), B=((1,), (2,)))
    plan = build_trie(forms)
    raw = plan.score({(): TokenScores({0: 0.0, 1: 0.0, 2: 0.0}, 2.0)}, context_token_count=0)

    assert plan.paths["A"] == (SpellingPath(("bare", "spaced"), (0,)),)
    assert _probabilities(raw) == pytest.approx({"A": 1 / 3, "B": 2 / 3})


def test_spellings_sharing_a_first_token_keep_its_mass_once_and_are_split_later():
    backend = DeclaredScores(
        {
            CONTEXT: {0: math.log(0.6), 3: math.log(0.3), 4: math.log(0.1)},
            (*CONTEXT, 0): {1: math.log(0.75), 2: math.log(0.25)},
        }
    )
    _, raw = score_with_backend(backend, CONTEXT, _forms(A=((0, 1), (0, 2)), B=((3,), (4,))))

    assert _probabilities(raw) == pytest.approx({"A": 0.6, "B": 0.4})
    assert [math.exp(path.logprob) for path in raw.scored_paths["A"]] == pytest.approx([0.45, 0.15])


def test_root_is_read_for_coverage_even_when_its_only_token_is_forced():
    forms = _forms(A=((0, 1), (0, 3)), B=((0, 2), (0, 4)))
    backend = DeclaredScores(
        {CONTEXT: {0: -math.inf}, (*CONTEXT, 0): dict.fromkeys(range(1, 5), 0.0)}
    )
    plan, raw = score_with_backend(backend, CONTEXT, forms)

    assert plan.readouts[0] == ReadoutRequest((), (0,))
    # The forced root ignores its score, so the -inf logit does not zero every credence.
    assert _probabilities(raw) == pytest.approx({"A": 0.5, "B": 0.5})
    assert raw.coverage == 0.0


def test_forced_prefixes_stay_in_later_contexts_without_their_own_queries():
    forms = _forms(A=((0, 1, 2), (4, 1, 2)), B=((0, 1, 3), (4, 1, 3)))
    backend = DeclaredScores(
        {
            CONTEXT: {0: 1.0, 4: 0.0},
            (*CONTEXT, 0, 1): {2: 0.0, 3: -math.inf},
            (*CONTEXT, 4, 1): {2: 0.0, 3: -math.inf},
        }
    )
    _, raw = score_with_backend(backend, CONTEXT, forms)

    assert backend.calls == [
        (CONTEXT, (0, 4)),
        ((*CONTEXT, 0, 1), (2, 3)),
        ((*CONTEXT, 4, 1), (2, 3)),
    ]
    assert raw.credence_logprobs == {"A": 0.0, "B": -math.inf}


def test_prefix_candidate_stops_only_on_distinct_end_tokens():
    forms = _forms(A=((0,), (3,)), AB=((0, 1), (3, 1)))
    fallback = {1: -math.inf, 10: -math.inf, 11: -math.inf}
    backend = DeclaredScores(
        {CONTEXT: {0: 0.0, 3: -1.0}, (*CONTEXT, 0): fallback, (*CONTEXT, 3): fallback}
    )
    plan, raw = score_with_backend(backend, CONTEXT, forms, end_tokens=[11, 10, 10])

    assert plan.end_tokens == (10, 11)
    assert backend.calls[1:] == [((*CONTEXT, 0), (1, 10, 11)), ((*CONTEXT, 3), (1, 10, 11))]
    assert _probabilities(raw) == pytest.approx({"A": 2 / 3, "AB": 1 / 3})


def test_leaf_candidates_do_not_query_end_scores():
    plan = build_trie(_forms(A=((0,), (2,)), B=((1,), (3,))), end_tokens=[10, 11])

    assert plan.readouts == (ReadoutRequest((), (0, 1, 2, 3)),)


def test_readouts_are_ordered_shortest_prefix_first_across_candidates():
    # The first candidate branches deep and the third shallow, so building
    # requests in candidate order alone would put (5, 6) before (0,).
    forms = _forms(A=((5, 6, 1), (9,)), B=((5, 6, 2), (8,)), C=((0, 1), (7,)), D=((0, 2), (4,)))

    assert [request.prefix for request in build_trie(forms).readouts] == [(), (0,), (5, 6)]


def test_missing_branch_context_or_score_remains_a_contract_error_even_at_zero_incoming_mass():
    plan = build_trie(_forms(A=((0, 1), (5,)), B=((0, 2), (6,)), C=((3,), (7,))))
    root = TokenScores({0: -math.inf, 3: 0.0, 5: -math.inf, 6: -math.inf, 7: -math.inf}, 1.0)

    with pytest.raises(BackendReadoutError, match="prefix"):
        plan.score({(): root}, context_token_count=0)
    with pytest.raises(BackendReadoutError, match="2"):
        plan.score({(): root, (0,): TokenScores({1: 0.0}, 1.0)}, context_token_count=0)
    raw = plan.score(
        {(): root, (0,): TokenScores({1: -math.inf, 2: -math.inf}, 1.0)}, context_token_count=0
    )
    assert raw.credence_logprobs == {"A": -math.inf, "B": -math.inf, "C": 0.0}


@pytest.mark.parametrize("bad_score", [math.nan, math.inf])
@pytest.mark.parametrize("token", [1, 10])
def test_zero_mass_descendants_still_validate_continuation_and_end_scores(bad_score, token):
    plan = build_trie(_forms(A=((0,), (5,)), AB=((0, 1), (6,)), C=((3,), (7,))), end_tokens=[10])
    branch = {1: 0.0, 10: 0.0}
    branch[token] = bad_score
    root = {0: -math.inf, 3: 0.0, 5: -math.inf, 6: -math.inf, 7: -math.inf}

    with pytest.raises(BackendReadoutError, match=str(token)):
        plan.score(
            {(): TokenScores(root, 1.0), (0,): TokenScores(branch, 1.0)}, context_token_count=0
        )


def test_extreme_float32_scores_survive_shared_roots_and_path_accumulation():
    maximum = float.fromhex("0x1.fffffep+127")  # Largest finite float32 value.
    forms = _forms(A=((4, 9), (0, 1)), B=((0, 2), (5,)), C=((3,), (6,)))
    backend = DeclaredScores(
        {
            CONTEXT: {0: maximum, 3: maximum, 4: maximum / 2, 5: -maximum, 6: -maximum},
            (*CONTEXT, 0): {1: -maximum, 2: maximum},
        }
    )
    _, raw = score_with_backend(backend, CONTEXT, forms)
    result = measurement_from_raw(raw)

    assert result.credences == {"A": 0.0, "B": 0.5, "C": 0.5}
    assert math.isfinite(result.raw.credence_logprobs["A"])
    assert result.top_labels == ("B", "C")


def test_cross_label_identical_paths_are_rejected():
    with pytest.raises(ValueError, match="owned"):
        build_trie(_forms(A=((0,), (1,)), B=((0,), (2,))))


@pytest.mark.parametrize("end_tokens", [(), (1,)])
def test_cross_label_prefix_requires_a_nonempty_disjoint_end_set(end_tokens):
    with pytest.raises(ValueError, match="END"):
        build_trie(_forms(A=((0,), (2,)), AB=((0, 1), (3,))), end_tokens=end_tokens)


@pytest.mark.parametrize("spellings", [((0,), (0, 1)), ((0, 1), (0,))])
def test_a_spelling_that_starts_the_same_candidates_other_spelling_is_rejected(spellings):
    with pytest.raises(ValueError, match="starts its other"):
        build_trie(_forms(A=spellings, B=((2,), (3,))))


@pytest.mark.parametrize("bad_path", [(), (-1,), (True,), (1.0,)])
def test_spelling_paths_require_nonempty_nonnegative_integer_token_ids(bad_path):
    with pytest.raises(ValueError):
        CandidatePaths(bad_path, (2,))


def test_bare_string_cannot_be_used_as_a_token_path():
    with pytest.raises(TypeError):
        CandidatePaths("1", (2,))


def test_candidates_must_be_given_as_named_spellings():
    with pytest.raises(TypeError, match="bare and spaced"):
        build_trie({"A": (0,), "B": CandidatePaths((1,), (2,))})


@pytest.mark.parametrize("end_tokens", [(-1,), (True,), (1.0,)])
def test_end_ids_require_nonnegative_integers(end_tokens):
    with pytest.raises(ValueError):
        build_trie(_forms(A=((0,), (2,)), B=((1,), (3,))), end_tokens=end_tokens)


@pytest.mark.parametrize("labels", [[], ["A"], ["", "B"]])
def test_plans_require_at_least_two_nonempty_candidate_labels(labels):
    with pytest.raises(ValueError):
        build_trie({label: CandidatePaths((i,), (i + 10,)) for i, label in enumerate(labels)})


def test_plan_owns_its_paths_and_is_reusable_without_semantic_state():
    bare, spaced, end_tokens = [0], [0, 1], [10]
    forms = {"A": CandidatePaths(bare, [2]), "AB": CandidatePaths(spaced, [3])}
    plan = build_trie(forms, end_tokens=end_tokens)
    bare.append(99)
    end_tokens.append(1)
    root = TokenScores({0: 0.0, 2: -math.inf, 3: -math.inf}, 1.0)

    def score(branch):
        return plan.score({(): root, (0,): TokenScores(branch, 1.0)}, context_token_count=0)

    first = score({1: 0.0, 10: 0.0})
    changed = score({1: -math.inf, 10: 0.0})
    assert plan.paths["A"][0].token_ids == (0,)
    assert plan.end_tokens == (10,)
    assert first.credence_logprobs == {"A": -math.log(2), "AB": -math.log(2)}
    assert changed.credence_logprobs == {"A": 0.0, "AB": -math.inf}
    assert score({1: 0.0, 10: 0.0}) == first


def test_long_forced_chain_is_not_limited_by_python_recursion_depth():
    prefix = (0,) * 1500
    plan = build_trie(_forms(A=(prefix + (1,), (2,)), B=(prefix + (3,), (4,))))
    root = TokenScores({0: 0.0, 2: -math.inf, 4: -math.inf}, 1.0)
    raw = plan.score({(): root, prefix: TokenScores({1: 0.0, 3: 0.0}, 1.0)}, context_token_count=0)

    assert [request.prefix for request in plan.readouts] == [(), prefix]
    assert raw.credence_logprobs == {"A": -math.log(2), "B": -math.log(2)}


def test_invalid_normalizer_at_a_later_readout_is_a_backend_error():
    plan = build_trie(_forms(A=((0, 1), (3,)), B=((0, 2), (4,))))
    root = TokenScores({0: 0.0, 3: -math.inf, 4: -math.inf}, 1.0)

    with pytest.raises(BackendReadoutError, match="normalizer"):
        plan.score({(): root, (0,): TokenScores({1: 0.0, 2: 0.0}, math.nan)}, context_token_count=0)
