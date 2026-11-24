"""Handles questions answered in a TUI, from an answers file, or one by one.

Every source goes through `run`, which applies each answer as soon as it is
given and records it if a recorder is configured.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace

import yaml


@dataclass(frozen=True)
class Link:
    label: str
    seconds: float
    url: str


@dataclass(frozen=True)
class Option:
    value: str
    title: str
    key: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A part of an answer that you can drop, such as a crowded candidate voice."""

    id: str
    title: str
    link: Link
    owner: str = ""
    """Person whose prominent video this candidate comes from.

    The candidate starts ticked when you choose that person.
    """


@dataclass
class Question:
    id: str
    lines: list[str]
    links: list[Link]
    options: list[Option]
    default: str | None = None
    prompt: str = "Who is this?"
    candidates: list[Candidate] = field(default_factory=list)
    pick_for: frozenset[str] = frozenset()
    """Answers that open the candidate picker in the TUI if there are candidates."""
    earlier: "Answer | None" = None
    """This session's earlier answer when the question is asked again."""
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    value: str
    drop: tuple[str, ...] = ()


SKIP = "skip"
QUIT = "quit"
LIST = "list"
"""A request to return to the question list."""

type Ask = Callable[[Question, int, int], Answer | str]
type Apply = Callable[[Question, Answer], object]
type Pick = Callable[[list[Question], dict[str, Answer]], int | None]
"""A callback returning the index to ask again, or `None` to finish."""


type Questions = list[Question] | Callable[[Mapping[str, Answer]], list[Question]]
"""A fixed list, or one that follows from the answers given so far."""


def run(
    questions: Questions,
    ask: Ask,
    apply: Apply,
    record: Apply | None = None,
    pick: Pick | None = None,
) -> list[tuple[Question, Answer]]:
    """Asks pending questions and offers the list through `pick` when available.

    After an answer, the next pending question comes up, bypassing questions
    skipped this session. Once none are left, `pick` can reopen a question
    or return `None` to finish. When a question is answered again, `apply`
    must replace the effect of the earlier answer. A callable `questions` is
    read again after every answer, so later questions can depend on it.
    """
    current = questions if callable(questions) else lambda answers: questions
    answers: dict[str, Answer] = {}
    skipped: set[str] = set()

    def chosen(listed: list[Question], index: int | None) -> str | None:
        return None if index is None else listed[index].id

    def following(qid: str) -> str | None:
        listed = current(answers)
        ids = [q.id for q in listed]
        start = ids.index(qid) if qid in ids else -1
        for step in range(1, len(listed) + 1):
            q = listed[(start + step) % len(listed)]
            if q.id not in answers and q.id not in skipped:
                return q.id
        return chosen(listed, pick(listed, answers)) if pick else None

    listed = current(answers)
    qid = listed[0].id if listed else None
    while qid is not None:
        listed = current(answers)
        index = [q.id for q in listed].index(qid)
        question = listed[index]
        if earlier := answers.get(question.id):
            question = replace(question, default=earlier.value, earlier=earlier)
        answer = ask(question, index, len(listed))
        if answer == QUIT:
            break
        if answer == LIST:
            qid = chosen(listed, pick(listed, answers)) if pick else following(qid)
            continue
        if answer == SKIP:
            skipped.add(question.id)
        else:
            if answer.value not in {o.value for o in question.options}:
                raise ValueError(f"{question.id}: {answer.value!r} is not an option")
            apply(question, answer)
            if record:
                record(question, answer)
            answers[question.id] = answer
        qid = following(qid)
    return [(q, answers[q.id]) for q in current(answers) if q.id in answers]


def from_answers(
    answers: Mapping[str, Answer],
    questions: list[Question],
    drops: Collection[str] = (),
) -> Ask:
    """Answers from a mapping and leaves questions with missing IDs pending."""
    ids = {q.id for q in questions}
    if unknown := sorted(set(answers) - ids):
        raise ValueError(f"no pending question: {', '.join(unknown)}")
    for q in questions:
        answer = answers.get(q.id)
        if answer and answer.value not in {o.value for o in q.options}:
            raise ValueError(f"{q.id}: {answer.value!r} is not an option")
    held = {c.id for q in questions for c in q.candidates}
    if stray := sorted(set(drops) - held):
        raise ValueError(f"no such candidate: {', '.join(stray)}")

    def ask(question: Question, index: int, total: int) -> Answer | str:
        answer = answers.get(question.id)
        if answer is None:
            return SKIP
        extra = tuple(c.id for c in question.candidates if c.id in drops)
        return Answer(answer.value, answer.drop + extra)

    return ask


def parse_answers(text: str) -> dict[str, Answer]:
    result = {}
    for qid, entry in (yaml.safe_load(text) or {}).items():
        if isinstance(entry, Mapping):
            result[str(qid)] = Answer(str(entry["value"]), tuple(entry["drop"]))
        else:
            result[str(qid)] = Answer(str(entry))
    return result


def dump_answers(answers: Mapping[str, Answer]) -> str:
    data = {
        qid: {"value": a.value, "drop": list(a.drop)} if a.drop else a.value
        for qid, a in answers.items()
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def parse_sets(pairs: list[str]) -> dict[str, Answer]:
    result = {}
    for pair in pairs:
        qid, sep, value = pair.rpartition("=")
        if not sep or not qid or not value:
            raise ValueError(f"{pair!r} is not ID=VALUE")
        result[qid] = Answer(value)
    return result
