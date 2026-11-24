"""Model-based tests: Hypothesis types keys into the real questionnaire.

Each machine runs `whd`'s questionnaire on a synthetic corpus, predicts
every screen it should show next, and checks the files after each step
against a model of the answers given so far.
"""

import os
import queue
import threading
import time
from collections import Counter
from typing import ClassVar

import e2e_world as world
import yaml
from hypothesis import HealthCheck, note, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from whodat.cli import questionnaire
from whodat.corpus import Clip
from whodat.labels import Label, format_time
from whodat.questionnaire import LIST, QUIT, SKIP, Answer
from whodat.tui import Tui
from whodat.workflows import NOBODY, Workspace

UP, DOWN, ENTER, SPACE, ESC, CTRL_C = "\x1b[A", "\x1b[B", "\r", " ", "\x1b", "\x03"
WAIT = 30.0


class Session:
    """One questionnaire run in a thread, reporting each screen it waits on."""

    def __init__(self, ws: Workspace, flow: str):
        self.events = queue.Queue()
        self.opened: list[str] = []
        self._pipe = create_pipe_input()
        self.pipe = self._pipe.__enter__()
        tui = Tui(
            self.opened.append,
            input=self.pipe,
            output=DummyOutput(),
            on_prompt=lambda screen, qid: self.events.put((screen, qid)),
        )
        if flow == "enroll":
            self.questions = ws.enrollment_questions()
            apply, name = ws.answer_voice, "enroll"
        else:
            self.questions = ws.review_questions("EVA")
            apply, name = (lambda q, a: ws.answer("EVA", q.id, a.value)), "review-EVA"

        def work():
            try:
                questionnaire(ws, name, self.questions, apply, None, False, tui=tui)
            except BaseException as error:  # noqa: BLE001 - reported to the test
                self.events.put(("error", error))
            else:
                self.events.put(("end", None))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def next(self) -> tuple[str, str | None]:
        event = self.events.get(timeout=WAIT)
        if event[0] == "error":
            raise event[1]
        return event

    def type(self, keys: str) -> None:
        self.pipe.send_text(keys)

    def close(self) -> None:
        """End the run with Ctrl-C, as a user would.

        Closing the pipe under a starting prompt leaves it polling a dead
        input forever; leftover threads starve the next examples.
        """
        while self.thread.is_alive():
            self.pipe.send_text(CTRL_C)
            self.thread.join(0.2)
        self._pipe.__exit__(None, None, None)


def moves(start: int, target: int) -> str:
    return DOWN * (target - start) if target > start else UP * (start - target)


class QuestionnaireMachine(RuleBasedStateMachine):
    flow = ""
    enrolled = False

    def __init__(self):
        super().__init__()
        self.root = world.fresh_root()
        os.environ["XDG_STATE_HOME"] = str(self.root / "state")
        corpus = world.build(self.root, enrolled=self.enrolled)
        self.ws = Workspace(corpus, "tone", embedders=world.EMBEDDERS)
        self.ever: dict[str, Answer] = {}
        self.all_ids: set[str] | None = None
        self.session = None
        self.start()

    def teardown(self):
        if self.session:
            self.session.close()
        world.remove(self.root)

    # Model of navigation

    def start(self):
        if self.session:
            self.session.close()
        self.session = Session(self.ws, self.flow)
        self.questions = self.session.questions
        ids = {q.id for q in self.questions}
        if self.all_ids is None:
            self.all_ids = ids
        assert ids == self.all_ids - set(self.ever), "pending questions"
        self.answers: dict[str, Answer] = {}
        self.skipped: set[str] = set()
        self.index = 0
        self.override = None
        self.expect(("who", self.questions[0].id) if self.questions else ("end", None))

    def expect(self, event):
        actual = self.session.next()
        assert actual == event, f"expected screen {event}, got {actual}"
        self.screen = event[0]

    @property
    def question(self):
        return self.questions[self.index]

    def who_choices(self) -> list:
        q = self.question
        return (
            [("open", link.url) for link in q.links]
            + [o.value for o in q.options]
            + [LIST, SKIP, QUIT]
        )

    def who_default(self):
        if self.override is not None:
            return self.override
        earlier = self.answers.get(self.question.id)
        return earlier.value if earlier else self.question.default

    def who_select(self, target) -> None:
        choices = self.who_choices()
        default = self.who_default()
        start = choices.index(default) if default in choices else 0
        self.session.type(moves(start, choices.index(target)) + ENTER)

    def following(self):
        n = len(self.questions)
        for step in range(1, n + 1):
            q = self.questions[(self.index + step) % n]
            if q.id not in self.answers and q.id not in self.skipped:
                self.index = (self.index + step) % n
                self.override = None
                return self.expect(("who", q.id))
        self.expect(("list", None))

    def commit(self, answer: Answer):
        self.answers[self.question.id] = answer
        self.ever[self.question.id] = answer
        self.following()

    def chose(self, value: str):
        q = self.question
        if value in q.pick_for and q.candidates:
            self.begin_candidates(value)
        else:
            self.commit(Answer(value))

    ACTIONS: ClassVar[dict[str, list[str]]] = {
        "who": [
            "name_by_number",
            "choose_by_arrows",
            "open_a_link",
            "skip",
            "back_to_the_list",
            "quit",
        ],
        "candidates": ["toggle", "listen", "confirm", "back_to_who"],
        "list": ["ask_again", "done"],
        "end": ["next_session"],
    }

    @rule(data=st.data())
    def step(self, data):
        """One user action, among those the current screen offers."""
        action = data.draw(st.sampled_from(self.ACTIONS[self.screen]), label="action")
        note(f"{self.screen}: {action}")
        getattr(self, action)(data)

    # Who is this?

    def name_by_number(self, data):
        option = data.draw(st.sampled_from([o for o in self.question.options if o.key]))
        self.session.type(option.key)
        self.chose(option.value)

    def choose_by_arrows(self, data):
        value = data.draw(st.sampled_from([o.value for o in self.question.options]))
        self.who_select(value)
        self.chose(value)

    def open_a_link(self, data):
        link = data.draw(st.sampled_from(self.question.links))
        self.who_select(("open", link.url))
        self.expect(("who", self.question.id))
        assert self.session.opened[-1] == link.url

    def skip(self, data):
        self.who_select(SKIP)
        self.skipped.add(self.question.id)
        self.following()

    def back_to_the_list(self, data):
        by_key = data.draw(st.booleans(), label="esc")
        if by_key:
            self.session.type(ESC)
        else:
            self.who_select(LIST)
        self.expect(("list", None))

    def quit(self, data):
        by_key = data.draw(st.booleans(), label="q")
        if by_key:
            self.session.type("q")
        else:
            self.who_select(QUIT)
        self.end()

    # Which candidates are this person?

    def begin_candidates(self, value: str):
        q = self.question
        own = [c for c in q.candidates if c.owner == value]
        self.rows = own + [c for c in q.candidates if c.owner != value]
        earlier = self.answers.get(q.id)
        if earlier and earlier.value == value:
            self.ticked = {c.id for c in q.candidates} - set(earlier.drop)
        else:
            self.ticked = {c.id for c in own} or {c.id for c in q.candidates}
        self.value, self.cursor = value, 0
        self.expect(("candidates", q.id))

    def toggle(self, data):
        row = data.draw(st.integers(0, len(self.rows) - 1))
        self.session.type(moves(self.cursor, row) + SPACE)
        self.cursor = row
        self.ticked ^= {self.rows[row].id}

    def listen(self, data):
        before = len(self.session.opened)
        self.session.type("o")
        deadline = time.time() + WAIT
        while len(self.session.opened) == before and time.time() < deadline:
            time.sleep(0.01)
        assert self.session.opened[-1] == self.rows[self.cursor].link.url

    def confirm(self, data):
        self.session.type(ENTER)
        q = self.question
        drop = tuple(c.id for c in q.candidates if c.id not in self.ticked)
        self.commit(Answer(self.value, drop))

    def back_to_who(self, data):
        self.session.type(ESC)
        self.override = self.value
        self.expect(("who", self.question.id))

    # The list of questions

    def ask_again(self, data):
        target = data.draw(st.integers(0, len(self.questions) - 1))
        self.session.type(moves(self.list_default(), target) + ENTER)
        self.index, self.override = target, None
        self.expect(("who", self.questions[target].id))

    def done(self, data):
        n = len(self.questions)
        self.session.type(moves(self.list_default(), n) + ENTER)
        self.end()

    def list_default(self) -> int:
        pending = [i for i, q in enumerate(self.questions) if q.id not in self.answers]
        return pending[0] if pending else len(self.questions)

    # Sessions

    def end(self):
        self.expect(("end", None))
        self.session.thread.join(WAIT)
        if self.answers:
            latest = max((self.ws.state / "answers").iterdir(), key=os.path.getmtime)
            recorded = yaml.safe_load(latest.read_text())
            assert recorded == {
                qid: {"value": a.value, "drop": list(a.drop)} if a.drop else a.value
                for qid, a in self.answers.items()
            }

    def next_session(self, data):
        self.start()


class EnrollMachine(QuestionnaireMachine):
    flow = "enroll"

    @invariant()
    def voices_match_the_answers(self):
        path = self.ws.state / "voices.json"
        recorded = yaml.safe_load(path.read_text()) if path.exists() else {}
        assert {qid: (v["value"], tuple(v["drop"])) for qid, v in recorded.items()} == {
            qid: (a.value, a.drop) for qid, a in self.ever.items()
        }

    @invariant()
    def clips_come_only_from_kept_candidates_of_that_person(self):
        candidates = self.ws.candidates()
        voices = {v["id"]: v["members"] for v in self.ws.voices()}
        allowed = {p: [] for p in self.ws.corpus.persons}
        for qid, answer in self.ever.items():
            if answer.value == NOBODY:
                continue
            for member in voices[qid]:
                if member not in answer.drop:
                    allowed[answer.value].append(candidates[member])
        for person, entry in self.ws.corpus.persons.items():
            for clip in entry.clips:
                assert any(_inside(clip, c) for c in allowed[person]), (person, clip)
            named = [q for q, a in self.ever.items() if a.value == person]
            kept = [m for q in named for m in voices[q] if m not in self.ever[q].drop]
            if len(named) == 1 and kept:
                assert entry.clips, f"{person} was named but has no clips"


class ReviewMachine(QuestionnaireMachine):
    flow = "review"
    enrolled = True

    @invariant()
    def labels_match_the_answers(self):
        expected = Counter(
            _rounded(Label(s, e, answer.value))
            for qid, answer in self.ever.items()
            for s, e in world.CLUSTERS[qid][1]
        )
        actual = Counter(_rounded(lb) for lb in self.ws.corpus.label_file("EVA").labels)
        assert actual == expected


def _inside(clip: Clip, candidate: dict) -> bool:
    return clip.video == candidate["video"] and any(
        c["start"] <= clip.start and clip.end <= c["end"] for c in candidate["clips"]
    )


def _rounded(label: Label) -> tuple:
    return format_time(label.start), format_time(label.end), label.identity


SETTINGS = settings(
    max_examples=int(os.environ.get("WHODAT_E2E_EXAMPLES", "20")),
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
TestEnrollQuestionnaire = EnrollMachine.TestCase
TestEnrollQuestionnaire.settings = SETTINGS
TestReviewQuestionnaire = ReviewMachine.TestCase
TestReviewQuestionnaire.settings = SETTINGS
