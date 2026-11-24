import math

from hypothesis import given
from hypothesis import strategies as st

from whodat.enroll import (
    DIFFERENT,
    SAME,
    cut_between,
    group_voices,
    kept_merges,
    merges,
    next_merge,
    suggested_merge,
)

# Two voices: rows 0-1 alike, rows 2-3 alike.
ROWS = [[1.0, 0.0], [0.95, 0.1], [0.0, 1.0], [0.1, 0.9]]


def test_merges_run_from_most_to_least_similar():
    ladder = merges(ROWS)

    assert [m.similarity for m in ladder] == sorted(
        (m.similarity for m in ladder), reverse=True
    )
    assert {frozenset(m.left + m.right) for m in ladder[:2]} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }
    assert sorted(ladder[2].left + ladder[2].right) == [0, 1, 2, 3]


def test_the_suggestion_is_the_first_merge_below_the_widest_gap():
    assert suggested_merge([0.95, 0.9, 0.3, 0.2]) == 2


def test_the_suggestion_ignores_wide_gaps_among_late_mixed_merges():
    # oneyplays with WeSpeaker: four people, so merges 20-22 join different ones.
    ladder = [0.97, 0.96, 0.96, 0.96, 0.95, 0.94, 0.94, 0.93, 0.93, 0.92, 0.91, 0.91]
    ladder += [0.89, 0.88, 0.87, 0.85, 0.84, 0.80, 0.77, 0.75, 0.52, 0.41, 0.12]

    assert suggested_merge(ladder) == 20


def test_a_cut_between_two_merges_keeps_only_those_above_it():
    sims = [m.similarity for m in merges(ROWS)]
    cut = cut_between(sims, 2)

    assert group_voices(ROWS, [1.0] * 4, cut) == [[0, 1], [2, 3]]


def test_same_above_different_settles_the_cut_between_them():
    answers = {1: SAME, 2: DIFFERENT}

    assert next_merge(4, answers, suggested=2) is None
    assert kept_merges(4, answers) == 2


def test_cant_tell_moves_the_question_elsewhere():
    assert next_merge(4, {2: "?"}, suggested=2) == 1


def test_unsettled_answers_keep_voices_apart():
    assert kept_merges(4, {0: SAME, 1: "?", 2: "?", 3: "?"}) == 1


def test_contradicting_answers_keep_voices_apart():
    assert kept_merges(4, {1: DIFFERENT, 2: SAME}) == 1


@given(
    count=st.integers(1, 40),
    data=st.data(),
)
def test_truthful_answers_find_the_cut_in_few_questions(count, data):
    truth = data.draw(st.integers(0, count), label="merges that are the same person")
    suggested = data.draw(st.integers(0, count - 1), label="suggested")
    answers = {}
    while (asked := next_merge(count, answers, suggested)) is not None:
        answers[asked] = SAME if asked < truth else DIFFERENT

    assert kept_merges(count, answers) == truth
    assert len(answers) <= math.ceil(math.log2(count + 1)) + 2
