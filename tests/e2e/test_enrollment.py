import json

import e2e_world as world
import pytest
from click.testing import CliRunner

from whodat import cli
from whodat.enroll import DIFFERENT, SAME
from whodat.questionnaire import Answer
from whodat.workflows import NOBODY, Workspace


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    corpus = world.build(tmp_path, enrolled=False)
    return Workspace(corpus, "tone", embedders=world.EMBEDDERS)


def voice(ws: Workspace, qid: str):
    return next(q for q in ws.enrollment_questions() if q.id == qid)


def pending(ws: Workspace) -> set[str]:
    return {q.id for q in ws.enrollment_questions()}


def clips(ws: Workspace) -> dict[str, int]:
    return {p: len(entry.clips) for p, entry in ws.corpus.persons.items()}


def test_naming_a_voice_again_moves_its_clips(ws):
    chris = voice(ws, "chris/g1")

    ws.answer_voice(chris, Answer("lyle"))
    ws.answer_voice(chris, Answer("chris"))

    assert clips(ws)["lyle"] == 0
    assert clips(ws)["chris"] > 0


def test_a_voice_answered_nobody_can_be_named_later(ws):
    chris = voice(ws, "chris/g1")

    ws.answer_voice(chris, Answer(NOBODY))
    ws.answer_voice(chris, Answer("lyle"))

    assert clips(ws)["lyle"] > 0


def answer_all_but_zach(ws: Workspace) -> None:
    for qid, value in (("chris/g1", "chris"), ("lyle/g1", "lyle"), ("lyle/g2", NOBODY)):
        ws.answer_voice(voice(ws, qid), Answer(value))


def test_reset_forgets_a_persons_clips_and_asks_their_voices_again(ws):
    answer_all_but_zach(ws)

    ws.reset_enrollment(["chris"])

    assert clips(ws)["chris"] == 0
    assert clips(ws)["lyle"] > 0
    assert pending(ws) == {"chris/g1", "zach/g2"}


def test_reset_without_persons_forgets_everyone(ws):
    answer_all_but_zach(ws)

    ws.reset_enrollment([])

    assert set(clips(ws).values()) == {0}
    assert pending(ws) == {"chris/g1", "zach/g2", "lyle/g1", "lyle/g2"}


def test_reset_of_an_unknown_person_fails(ws):
    with pytest.raises(ValueError, match="nobody"):
        ws.reset_enrollment(["nobody"])


def test_reset_asks_again_a_voice_holding_the_persons_candidates(ws):
    ws.answer_voice(voice(ws, "chris/g1"), Answer(NOBODY))

    ws.reset_enrollment(["zach"])

    assert "chris/g1" in pending(ws)


SPEAKER = {
    f"{person}/g{i + 1}": speaker
    for person, windows in world.CANDIDATES.items()
    for i, (_, _, speaker) in enumerate(windows)
}


def answer_cut_truthfully(ws: Workspace) -> int:
    asked = 0
    while questions := ws.cut_questions({}):
        (q,) = questions
        sides = q.details["left"] + q.details["right"]
        same = len({SPEAKER[m] for m in sides}) == 1
        ws.answer_cut(q, Answer(SAME if same else DIFFERENT))
        asked += 1
    return asked


@pytest.fixture
def uncut(ws):
    ws.reset_voice_cut()
    return ws


def test_voices_wait_for_the_cut(uncut):
    with pytest.raises(ValueError, match="voice cut"):
        uncut.enrollment_questions()


def test_truthful_answers_settle_a_cut_that_splits_the_speakers(uncut):
    asked = answer_cut_truthfully(uncut)

    groups = {frozenset(v["members"]) for v in uncut.voices()}
    assert groups == {
        frozenset({"chris/g1", "chris/g2", "zach/g1"}),
        frozenset({"zach/g2", "zach/g3"}),
        frozenset({"lyle/g1"}),
        frozenset({"lyle/g2"}),
    }
    assert asked <= 4
    assert uncut.voice_cut() == uncut.corpus.voice_cut("tone")


def answer_three_voices(ws: Workspace) -> None:
    for qid, value in (("chris/g1", "chris"), ("lyle/g1", "lyle"), ("zach/g2", "zach")):
        ws.answer_voice(voice(ws, qid), Answer(value))


def test_a_new_cut_that_keeps_the_voices_keeps_their_answers(ws):
    answer_three_voices(ws)

    ws.corpus.save_voice_cut("tone", 0.5)

    assert pending(ws) == {"lyle/g2"}
    assert set(clips(ws).values()) != {0}


def test_a_new_cut_that_regroups_voices_asks_about_them_again(ws):
    answer_three_voices(ws)

    ws.corpus.save_voice_cut("tone", -0.5)

    assert pending(ws) == {"chris/g1"}
    assert set(clips(ws).values()) == {0}


def test_resetting_the_cut_asks_for_it_again(ws):
    ws.reset_voice_cut()

    assert ws.needs_voice_cut()
    assert len(ws.cut_questions({})) == 1


def whd_enroll(ws: Workspace, *args: str):
    result = CliRunner().invoke(cli.enroll, args, obj=ws, catch_exceptions=False)
    return json.loads(result.stdout)


def test_scripts_settle_the_cut_then_name_voices(uncut):
    while uncut.needs_voice_cut():
        (listed,) = whd_enroll(uncut, "review", "-l")
        sides = listed["details"]["left"] + listed["details"]["right"]
        same = len({SPEAKER[m] for m in sides}) == 1
        whd_enroll(
            uncut, "review", "-s", f"{listed['id']}={SAME if same else DIFFERENT}"
        )

    listed = whd_enroll(uncut, "review", "-l")

    assert {q["id"] for q in listed} == {"chris/g1", "zach/g2", "lyle/g1", "lyle/g2"}
