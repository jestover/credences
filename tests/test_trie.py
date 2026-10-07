"""Selected-path ownership, stopping boundaries, and exact readout requirements."""

import math
from dataclasses import FrozenInstanceError

import pytest

from credences import RawReadout
from credences.errors import BackendReadoutError
from credences.probabilities import measurement_from_raw
from credences.trie import (
    CandidatePaths,
    build_trie,
    first_token_ids,
    select_paths,
    validate_potential_paths,
)


class DeclaredLogits:
    """Record requests and return only declared scores at exact token contexts."""

    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def next_token_logits(self, token_ids, wanted):
        context = tuple(token_ids)
        self.calls.append((context, tuple(wanted)))
        assert context in self.tables, f"Unexpected readout at {context}"
        return {
            token: self.tables[context][token] for token in wanted if token in self.tables[context]
        }


def _run(forms, backend, *, context=(90, 91), end_tokens=()):
    """The test caller drives scores; the production plan only consumes them."""
    validate_potential_paths(forms, end_tokens=end_tokens)
    first = backend.next_token_logits(context, first_token_ids(forms))
    selected = select_paths(forms, first)
    plan = build_trie(selected, end_tokens=end_tokens)
    readouts = {(): first}
    for request in plan.additional_readouts:
        readouts[request.prefix] = backend.next_token_logits(
            context + request.prefix, request.wanted
        )
    return selected, plan, plan.score(readouts)


@pytest.mark.parametrize("score", [2.0, -math.inf])
def test_exact_form_ties_prefer_named_bare_path_and_preserve_its_suffix(score):
    forms = {
        "B": CandidatePaths(spaced=(4,), bare=(0, 2, 3)),
        "A": CandidatePaths(spaced=(5,), bare=(0, 1)),
    }
    selected = select_paths(forms, {0: score, 4: score, 5: score})

    assert selected == {"B": (0, 2, 3), "A": (0, 1)}
    assert list(selected) == ["B", "A"]


def test_spaced_selection_keeps_its_entire_different_suffix():
    forms = {
        "A": CandidatePaths(bare=(0, 1), spaced=(4, 8, 9)),
        "B": CandidatePaths(bare=(2,), spaced=(5,)),
    }

    assert select_paths(forms, {0: 0.0, 2: 0.0, 4: 1.0, 5: -1.0}) == {
        "A": (4, 8, 9),
        "B": (2,),
    }


def test_same_first_id_is_requested_once_but_remains_a_selection_readout():
    forms = {
        "A": CandidatePaths(bare=(0, 1), spaced=(0, 3)),
        "B": CandidatePaths(bare=(0, 2), spaced=(0, 4)),
    }
    backend = DeclaredLogits({(90, 91): {0: -math.inf}, (90, 91, 0): {1: 0.0, 2: 0.0}})
    _, plan, logs = _run(forms, backend)

    assert first_token_ids(forms) == (0,)
    assert [request.prefix for request in plan.readouts] == [(0,)]
    assert backend.calls == [((90, 91), (0,)), ((90, 91, 0), (1, 2))]
    assert logs == {"A": -math.log(2), "B": -math.log(2)}


def test_selection_validates_all_requested_first_scores_even_when_selected_root_is_forced():
    forms = {
        "A": CandidatePaths((0, 1), (4,)),
        "B": CandidatePaths((0, 2), (5,)),
    }
    with pytest.raises(BackendReadoutError, match="5"):
        select_paths(forms, {0: 1.0, 4: 0.0})
    with pytest.raises(BackendReadoutError, match="5"):
        select_paths(forms, {0: 1.0, 4: 0.0, 5: math.nan})


def test_selected_root_reuses_initial_scores_and_ignores_discarded_forms():
    forms = {
        "positive": CandidatePaths((0, 1), (4,)),
        "position": CandidatePaths((0, 2, 3), (5,)),
    }
    backend = DeclaredLogits({(90, 91): {0: math.log(0.3), 4: math.log(0.4), 5: math.log(0.2)}})
    selected, plan, logs = _run(forms, backend)

    assert selected == {"positive": (4,), "position": (0, 2, 3)}
    assert plan.additional_readouts == ()
    assert plan.readouts[0].wanted == (0, 4)
    assert backend.calls == [((90, 91), (0, 4, 5))]
    assert {label: math.exp(value) for label, value in logs.items()} == pytest.approx(
        {"positive": 4 / 7, "position": 3 / 7}
    )


def test_forced_prefixes_stay_in_later_contexts_without_their_own_queries():
    forms = {
        "A": CandidatePaths((0, 1, 2), (4, 1, 2)),
        "B": CandidatePaths((0, 1, 3), (4, 1, 3)),
    }
    backend = DeclaredLogits({(90, 91): {0: 1.0, 4: 0.0}, (90, 91, 0, 1): {2: 0.0, 3: -math.inf}})
    _, plan, logs = _run(forms, backend)

    assert [request.prefix for request in plan.additional_readouts] == [(0, 1)]
    assert backend.calls == [((90, 91), (0, 4)), ((90, 91, 0, 1), (2, 3))]
    assert logs == {"A": 0.0, "B": -math.inf}


def test_shared_root_weight_is_counted_once_in_five_candidate_golden_example():
    forms = {
        "very negative": CandidatePaths((0, 1), (6, 1)),
        "negative": CandidatePaths((2,), (7,)),
        "neutral": CandidatePaths((3,), (8,)),
        "positive": CandidatePaths((4,), (9,)),
        "very positive": CandidatePaths((0, 5), (6, 5)),
    }
    backend = DeclaredLogits(
        {
            (90, 91): {
                token: math.log(weight)
                for token, weight in {
                    0: 0.10,
                    2: 0.65,
                    3: 0.20,
                    4: 0.02,
                    6: 0.001,
                    7: 0.002,
                    8: 0.001,
                    9: 0.001,
                }.items()
            },
            (90, 91, 0): {1: math.log(0.95), 5: math.log(0.01)},
        }
    )
    _, _, logs = _run(forms, backend)

    assert {label: math.exp(value) for label, value in logs.items()} == pytest.approx(
        {
            "very negative": (0.10 / 0.97) * (0.95 / 0.96),
            "negative": 0.65 / 0.97,
            "neutral": 0.20 / 0.97,
            "positive": 0.02 / 0.97,
            "very positive": (0.10 / 0.97) * (0.01 / 0.96),
        }
    )
    assert len(backend.calls) == 2


def test_prefix_candidate_stops_only_on_distinct_end_tokens():
    forms = {"A": CandidatePaths((0,), (3,)), "AB": CandidatePaths((0, 1), (3, 1))}
    backend = DeclaredLogits(
        {(90, 91): {0: 0.0, 3: -1.0}, (90, 91, 0): {1: -math.inf, 10: -math.inf, 11: -math.inf}}
    )
    _, plan, logs = _run(forms, backend, end_tokens=[11, 10, 10])

    assert plan.end_tokens == (10, 11)
    assert backend.calls == [((90, 91), (0, 3)), ((90, 91, 0), (1, 10, 11))]
    assert {label: math.exp(value) for label, value in logs.items()} == pytest.approx(
        {"A": 2 / 3, "AB": 1 / 3}
    )


def test_leaf_candidates_do_not_query_end_scores():
    plan = build_trie({"A": (0,), "B": (1,)}, end_tokens=[10, 11])

    assert [request.prefix for request in plan.readouts] == [()]
    assert plan.additional_readouts == ()
    assert plan.score({(): {0: 0.0, 1: 0.0}}) == {"A": -math.log(2), "B": -math.log(2)}


def test_missing_branch_context_or_score_remains_a_contract_error_even_at_zero_incoming_mass():
    plan = build_trie({"A": (0, 1), "B": (0, 2), "C": (3,)})
    root = {0: -math.inf, 3: 0.0}

    with pytest.raises(BackendReadoutError, match="prefix"):
        plan.score({(): root})
    with pytest.raises(BackendReadoutError, match="2"):
        plan.score({(): root, (0,): {1: 0.0}})
    assert plan.score({(): root, (0,): {1: -math.inf, 2: -math.inf}}) == {
        "A": -math.inf,
        "B": -math.inf,
        "C": 0.0,
    }


@pytest.mark.parametrize("bad_score", [math.nan, math.inf])
@pytest.mark.parametrize("token", [1, 10])
def test_zero_mass_descendants_still_validate_continuation_and_end_scores(bad_score, token):
    plan = build_trie({"A": (0,), "AB": (0, 1), "C": (3,)}, end_tokens=[10])
    branch = {1: 0.0, 10: 0.0}
    branch[token] = bad_score

    with pytest.raises(BackendReadoutError, match=str(token)):
        plan.score({(): {0: -math.inf, 3: 0.0}, (0,): branch})


def test_extreme_float32_scores_survive_selection_shared_roots_and_path_accumulation():
    maximum = float.fromhex("0x1.fffffep+127")  # Largest finite float32 value.
    forms = {
        "A": CandidatePaths(bare=(4, 9), spaced=(0, 1)),
        "B": CandidatePaths(bare=(0, 2), spaced=(5,)),
        "C": CandidatePaths(bare=(3,), spaced=(6,)),
    }
    backend = DeclaredLogits(
        {
            (90, 91): {0: maximum, 3: maximum, 4: maximum / 2, 5: 0.0, 6: 0.0},
            (90, 91, 0): {1: -maximum, 2: maximum},
        }
    )
    selected, _, logs = _run(forms, backend)
    result = measurement_from_raw(RawReadout(logs, selected, 2))

    assert selected == {"A": (0, 1), "B": (0, 2), "C": (3,)}
    assert result.credences == {"A": 0.0, "B": 0.5, "C": 0.5}
    assert math.isfinite(result.raw.credence_logprobs["A"])
    assert result.raw.credence_logprobs["A"] == -2 * maximum
    assert result.top_labels == ("B", "C")
    assert backend.calls == [((90, 91), (0, 3, 4, 5, 6)), ((90, 91, 0), (1, 2))]


@pytest.mark.parametrize("paths", [{"A": (), "B": (1,)}, {"A": (0,), "B": (0,)}])
def test_empty_or_cross_label_identical_selected_paths_are_rejected(paths):
    with pytest.raises(ValueError):
        build_trie(paths)


@pytest.mark.parametrize("end_tokens", [(), (1,)])
def test_selected_prefix_overlap_requires_a_nonempty_disjoint_end_set(end_tokens):
    with pytest.raises(ValueError, match="END"):
        build_trie({"A": (0,), "AB": (0, 1)}, end_tokens=end_tokens)


@pytest.mark.parametrize("end_tokens", [(), (1,)])
def test_potential_prefix_overlap_is_rejected_before_scores_even_if_selection_could_avoid_it(
    end_tokens,
):
    forms = {"A": CandidatePaths((0,), (2,)), "AB": CandidatePaths((0, 1), (3,))}
    backend = DeclaredLogits({(90, 91): {0: 0.0, 2: 2.0, 3: 2.0}})

    with pytest.raises(ValueError, match="END"):
        _run(forms, backend, end_tokens=end_tokens)
    assert backend.calls == []
    # With already-selected disjoint paths, no stopping boundary needs END.
    assert build_trie({"A": (2,), "AB": (3,)}).end_tokens == ()


def test_cross_label_alternative_path_collision_is_rejected_before_scores():
    forms = {"A": CandidatePaths((0,), (2,)), "B": CandidatePaths((1,), (2,))}
    backend = DeclaredLogits({})

    with pytest.raises(ValueError, match="owned"):
        _run(forms, backend)
    assert backend.calls == []


def test_same_candidate_alternatives_do_not_create_stopping_boundaries():
    forms = {"A": CandidatePaths((0,), (0, 1)), "B": CandidatePaths((2,), (3,))}
    validate_potential_paths(forms)
    validate_potential_paths(forms, end_tokens=[1])

    assert select_paths(forms, {0: 0.0, 2: 0.0, 3: 0.0}) == {"A": (0,), "B": (2,)}


def test_same_candidate_identical_alternatives_are_allowed_and_requested_once():
    forms = {"A": CandidatePaths((0,), (0,)), "B": CandidatePaths((1,), (2,))}
    validate_potential_paths(forms)

    assert first_token_ids(forms) == (0, 1, 2)
    assert select_paths(forms, {0: 0.0, 1: 0.0, 2: -1.0}) == {"A": (0,), "B": (1,)}


@pytest.mark.parametrize("bad_path", [(), (-1,), (True,), (1.0,)])
def test_alternative_paths_require_nonempty_nonnegative_integer_token_ids(bad_path):
    with pytest.raises(ValueError):
        CandidatePaths(bad_path, (2,))


def test_bare_string_cannot_be_used_as_a_token_path():
    with pytest.raises(TypeError):
        CandidatePaths("1", (2,))


@pytest.mark.parametrize("end_tokens", [(-1,), (True,), (1.0,)])
def test_end_ids_require_nonnegative_integers(end_tokens):
    with pytest.raises(ValueError):
        build_trie({"A": (0,), "B": (1,)}, end_tokens=end_tokens)


@pytest.mark.parametrize("paths", [{}, {"A": (0,)}, {"": (0,), "B": (1,)}])
def test_plans_require_at_least_two_nonempty_candidate_labels(paths):
    with pytest.raises(ValueError):
        build_trie(paths)


def test_plan_owns_immutable_paths_and_is_reusable_without_semantic_state():
    paths = {"A": [0], "AB": [0, 1]}
    end_tokens = [10]
    plan = build_trie(paths, end_tokens=end_tokens)
    paths["A"].append(99)
    paths["AB"] = [2]
    end_tokens.append(1)

    assert plan.paths == (("A", (0,)), ("AB", (0, 1)))
    assert plan.end_tokens == (10,)
    with pytest.raises(FrozenInstanceError):
        plan.end_tokens = ()
    with pytest.raises(FrozenInstanceError):
        plan.readouts[0].wanted = ()
    first = plan.score({(0,): {1: 0.0, 10: 0.0}})
    changed = plan.score({(0,): {1: -math.inf, 10: 0.0}})
    assert first == {"A": -math.log(2), "AB": -math.log(2)}
    assert changed == {"A": 0.0, "AB": -math.inf}
    assert plan.score({(0,): {1: 0.0, 10: 0.0}}) == first


def test_alternative_paths_own_their_input_lists():
    bare = [0, 1]
    spaced = [2]
    paths = CandidatePaths(bare, spaced)
    bare.append(99)
    spaced.append(99)

    assert paths.bare == (0, 1)
    assert paths.spaced == (2,)


def test_long_forced_chain_is_not_limited_by_python_recursion_depth():
    prefix = (0,) * 1500
    plan = build_trie({"A": prefix + (1,), "B": prefix + (2,)})

    assert len(plan.readouts) == 1
    assert plan.readouts[0].prefix == prefix
    assert plan.score({prefix: {1: 0.0, 2: 0.0}}) == {"A": -math.log(2), "B": -math.log(2)}
