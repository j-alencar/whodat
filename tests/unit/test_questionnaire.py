import pytest

from whodat.questionnaire import (
    LIST,
    QUIT,
    SKIP,
    Answer,
    Candidate,
    Link,
    Option,
    Question,
    dump_answers,
    from_answers,
    parse_answers,
    parse_sets,
    run,
)

LINK = Link("1:09:00", 7.1, "https://youtu.be/v?t=4140")
OPTIONS = [Option("chris", "Chris", "1"), Option("lyle", "Lyle", "2")]


def question(qid: str, candidates=()) -> Question:
    return Question(
        qid,
        lines=[qid],
        links=[LINK],
        options=OPTIONS,
        default="chris",
        candidates=list(candidates),
        pick_for=frozenset({"chris", "lyle"}),
    )


def candidate(cid: str) -> Candidate:
    return Candidate(cid, cid, LINK)


def test_answers_apply_in_question_order_and_missing_ones_stay_pending():
    questions = [question("a"), question("b"), question("c")]
    applied = []

    run(
        questions,
        from_answers({"c": Answer("lyle"), "a": Answer("chris")}, questions),
        lambda q, a: applied.append((q.id, a.value)),
    )

    assert applied == [("a", "chris"), ("c", "lyle")]


def test_unknown_ids_fail_before_anything_applies():
    questions = [question("a")]

    with pytest.raises(ValueError, match="zz"):
        from_answers({"a": Answer("chris"), "zz": Answer("lyle")}, questions)


def test_an_answer_outside_the_options_fails_before_anything_applies():
    questions = [question("a"), question("b")]

    with pytest.raises(ValueError, match="zach"):
        from_answers({"a": Answer("chris"), "b": Answer("zach")}, questions)


def test_drops_attach_to_the_question_holding_that_candidate():
    questions = [
        question("a", [candidate("chris/g1"), candidate("chris/g2")]),
        question("b", [candidate("zach/g1")]),
    ]
    applied = {}

    run(
        questions,
        from_answers(
            {"a": Answer("chris"), "b": Answer("lyle")}, questions, drops={"chris/g2"}
        ),
        lambda q, a: applied.update({q.id: a.drop}),
    )

    assert applied == {"a": ("chris/g2",), "b": ()}


def test_a_drop_outside_every_question_fails():
    questions = [question("a", [candidate("chris/g1")])]

    with pytest.raises(ValueError, match="chris/g9"):
        from_answers({"a": Answer("chris")}, questions, drops={"chris/g9"})


def test_each_applied_answer_is_recorded_as_it_happens():
    questions = [question("a"), question("b")]
    events = []

    run(
        questions,
        from_answers({"a": Answer("chris"), "b": Answer("lyle")}, questions),
        lambda q, a: events.append(("apply", q.id)),
        record=lambda q, a: events.append(("record", q.id)),
    )

    assert events == [
        ("apply", "a"),
        ("record", "a"),
        ("apply", "b"),
        ("record", "b"),
    ]


def test_answers_files_round_trip():
    answers = {
        "1200s/SPEAKER_02": Answer("chris"),
        "1200s/SPEAKER_00": Answer("?"),
        "zach/g1": Answer("tomar", ("tomar/g3",)),
    }

    assert parse_answers(dump_answers(answers)) == answers


def test_sets_parse_into_answers():
    assert parse_sets(["1200s/SPEAKER_02=chris", "x=?"]) == {
        "1200s/SPEAKER_02": Answer("chris"),
        "x": Answer("?"),
    }


def test_a_set_without_a_value_fails():
    with pytest.raises(ValueError, match="ID=VALUE"):
        parse_sets(["1200s/SPEAKER_02"])


def scripted(*steps):
    """An ask that replays `steps` and records the defaults it was shown."""
    steps = list(steps)
    shown = []

    def ask(q, index, total):
        shown.append((q.id, q.default))
        return steps.pop(0)

    return ask, shown


def test_after_an_answer_from_the_list_the_next_pending_question_follows():
    questions = [question("a"), question("b"), question("c")]
    ask, shown = scripted(LIST, Answer("lyle"), Answer("chris"), Answer("chris"))
    picks = iter([2, None])
    applied = []

    run(
        questions,
        ask,
        lambda q, a: applied.append(q.id),
        pick=lambda qs, answers: next(picks),
    )

    assert [qid for qid, _ in shown] == ["a", "c", "a", "b"]
    assert applied == ["c", "a", "b"]


def test_reanswering_applies_the_new_answer_last():
    questions = [question("a"), question("b")]
    ask, _ = scripted(Answer("lyle"), LIST, Answer("chris"), Answer("lyle"))
    picks = iter([0, None])
    applied = []

    done = run(
        questions,
        ask,
        lambda q, a: applied.append((q.id, a.value)),
        pick=lambda qs, answers: next(picks),
    )

    assert applied == [("a", "lyle"), ("a", "chris"), ("b", "lyle")]
    assert [(q.id, a.value) for q, a in done] == [("a", "chris"), ("b", "lyle")]


def test_skipped_questions_wait_for_the_list():
    questions = [question("a"), question("b")]
    ask, shown = scripted(SKIP, Answer("lyle"))
    listed = []

    run(questions, ask, lambda q, a: None, pick=lambda qs, a: listed.append(1))

    assert [qid for qid, _ in shown] == ["a", "b"]
    assert listed == [1]


def test_a_reasked_question_starts_on_its_previous_answer():
    questions = [question("a")]
    ask, shown = scripted(Answer("lyle"), QUIT)
    picks = iter([0])

    run(questions, ask, lambda q, a: None, pick=lambda qs, answers: next(picks))

    assert shown == [("a", "chris"), ("a", "lyle")]


def test_the_list_shows_the_answers_so_far():
    questions = [question("a"), question("b")]
    ask, _ = scripted(Answer("lyle"), LIST)
    seen = []

    def pick(qs, answers):
        seen.append(dict(answers))

    run(questions, ask, lambda q, a: None, pick=pick)

    assert seen == [{"a": Answer("lyle")}]


def test_without_a_list_the_run_ends_after_the_last_question():
    questions = [question("a")]
    ask, _ = scripted(Answer("lyle"))

    done = run(questions, ask, lambda q, a: None)

    assert len(done) == 1


def test_questions_can_follow_from_earlier_answers():
    def questions(answers):
        asked = [question("a")]
        if "a" in answers:
            asked.append(question(f"after-{answers['a'].value}"))
        return asked

    ask, shown = scripted(Answer("lyle"), Answer("chris"))

    done = run(questions, ask, lambda q, a: None)

    assert [qid for qid, _ in shown] == ["a", "after-lyle"]
    assert [q.id for q, _ in done] == ["a", "after-lyle"]
