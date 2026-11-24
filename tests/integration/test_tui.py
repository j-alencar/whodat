import io
import threading
import time
from dataclasses import replace

from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.renderer import Renderer

from whodat.questionnaire import (
    LIST,
    QUIT,
    SKIP,
    Answer,
    Candidate,
    Link,
    Option,
    Question,
)
from whodat.tui import Tui

UP, DOWN, ENTER, SPACE, ESC, CTRL_C = "\x1b[A", "\x1b[B", "\r", " ", "\x1b", "\x03"
LINK = Link("1:09:00", 7.1, "https://youtu.be/v?t=4140")


def question(candidates=()) -> Question:
    return Question(
        "1200s/SPEAKER_02",
        lines=["Review"],
        links=[LINK],
        options=[
            Option("chris", "Chris", "1"),
            Option("lyle", "Lyle", "2"),
            Option("unknown", "Someone not in the roster"),
        ],
        default="lyle",
        candidates=list(candidates),
        pick_for=frozenset({"chris", "lyle"}),
    )


def answer(question: Question, *keys: str, open_url=None):
    """Types each string of keys into its own prompt, in order."""
    return typed(lambda tui: tui(question, 0, 1), keys, open_url)


def typed(call, keys, open_url=None, output=None):
    with create_pipe_input() as pipe:
        tui = Tui(open_url, input=pipe, output=output or DummyOutput())

        def typist():
            for k in keys:
                time.sleep(0.3)  # let the prompt start
                pipe.send_text(k)

        thread = threading.Thread(target=typist)
        thread.start()
        result = call(tui)
        thread.join()
        return result


def test_a_roster_number_answers_at_once():
    assert answer(question(), "1") == Answer("chris")


def test_the_cursor_starts_on_the_default():
    assert answer(question(), ENTER) == Answer("lyle")


def test_options_after_the_roster_are_reachable_by_arrows():
    assert answer(question(), DOWN + ENTER) == Answer("unknown")


def test_the_list_of_questions_follows_the_options():
    assert answer(question(), DOWN * 2 + ENTER) == LIST


def test_skip_follows_the_list():
    assert answer(question(), DOWN * 3 + ENTER) == SKIP


def test_ctrl_c_quits():
    assert answer(question(), CTRL_C) == QUIT


def test_open_choices_open_the_link_then_ask_again():
    opened = []

    result = answer(question(), UP * 2 + ENTER, "2", open_url=opened.append)

    assert opened == [LINK.url]
    assert result == Answer("lyle")


CANDIDATES = [
    Candidate("chris/g1", "chris/g1", Link("1:00", 9.0, "https://c1"), "chris"),
    Candidate("chris/g2", "chris/g2", Link("2:00", 9.0, "https://c2"), "chris"),
    Candidate("zach/g1", "zach/g1", Link("3:00", 9.0, "https://z1"), "zach"),
]


def test_only_the_named_persons_own_candidates_start_ticked():
    assert answer(question(CANDIDATES), "1", ENTER) == Answer("chris", ("zach/g1",))


def test_space_unticks_the_highlighted_candidate():
    result = answer(question(CANDIDATES), "1", SPACE + ENTER)

    assert result == Answer("chris", ("chris/g1", "zach/g1"))


def test_o_opens_the_highlighted_candidate_and_stays():
    opened = []

    result = answer(
        question(CANDIDATES), "1", DOWN + "o", ENTER, open_url=opened.append
    )

    assert opened == ["https://c2"]
    assert result == Answer("chris", ("zach/g1",))


def test_esc_goes_back_to_who_is_this():
    assert answer(question(CANDIDATES), "1", ESC, "2", ENTER) == Answer("lyle")


def test_with_no_own_candidates_all_start_ticked():
    assert answer(question(CANDIDATES), "2", ENTER) == Answer("lyle")


def test_asked_again_with_the_same_name_keeps_the_earlier_ticks():
    earlier = Answer("chris", ("chris/g2",))
    asked = replace(question(CANDIDATES), default="chris", earlier=earlier)

    assert answer(asked, "1", ENTER) == earlier


def test_esc_on_who_is_this_opens_the_list():
    assert answer(question(), ESC) == LIST


def test_q_quits():
    assert answer(question(), "q") == QUIT


def test_the_list_starts_on_the_first_pending_question():
    questions = [replace(question(), id=qid) for qid in ("a", "b", "c")]
    answers = {"a": Answer("chris")}

    assert typed(lambda tui: tui.pick(questions, answers), [ENTER]) == 1


def test_the_list_starts_on_done_when_all_are_answered():
    questions = [replace(question(), id=qid) for qid in ("a", "b")]
    answers = {"a": Answer("chris"), "b": Answer("lyle")}

    assert typed(lambda tui: tui.pick(questions, answers), [ENTER]) is None
    assert typed(lambda tui: tui.pick(questions, answers), [UP * 2 + ENTER]) == 0


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_terminal_that_never_reports_its_cursor_gets_no_warning(monkeypatch):
    monkeypatch.setattr(Renderer, "CPR_TIMEOUT", 0.05)
    screen = Terminal()
    output = Vt100_Output(screen, lambda: Size(24, 80), term="xterm")
    typed(lambda tui: tui(question(), 0, 1), [ENTER], output=output)
    assert "WARNING" not in screen.getvalue()
