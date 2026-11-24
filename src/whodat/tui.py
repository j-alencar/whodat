"""Collects questionnaire answers in the terminal."""

import os
import shlex
import shutil
import subprocess
import webbrowser
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import click
import questionary
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from questionary.prompts.common import InquirerControl

from whodat.questionnaire import LIST, QUIT, SKIP, Answer, Link, Question

type Open = Callable[[str], object]

DONE = "done"
ESC_WAIT = 0.05
"""Seconds to distinguish a lone Escape from an arrow key sequence.

The `prompt_toolkit` default is 0.5 seconds.
"""
BACK = "back"


def browser() -> Open | None:
    """Returns a URL opener for the configured browser, or `None` if unavailable."""
    if command := os.environ.get("BROWSER"):
        return partial(_launch, shlex.split(command.split(os.pathsep)[0]))
    if shutil.which("xdg-mime"):
        handler = subprocess.run(
            ["xdg-mime", "query", "default", "x-scheme-handler/https"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return partial(_launch, ["xdg-open"]) if handler else None
    try:
        webbrowser.get()
    except webbrowser.Error:
        return None
    return webbrowser.open


def _launch(command: list[str], url: str) -> None:
    """Detaches the browser so it cannot read the prompt's terminal input."""
    args = [arg.replace("%s", url) for arg in command]
    if not any("%s" in arg for arg in command):
        args.append(url)
    subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def link_line(link: Link) -> str:
    return f"{link.label}  {link.seconds:.1f} s  {link.url}"


def open_title(link: Link) -> str:
    return f"↗ Open {link.label} ({link.seconds:.1f} s)"


class Tui:
    """Uses `questionary` to ask each question.

    Passes `input` and `output` to `prompt_toolkit` so tests can send keystrokes.
    """

    def __init__(
        self, open_url: Open | None = None, input=None, output=None, on_prompt=None
    ):
        self.open_url = open_url
        self.io = {"input": input, "output": output} if input else {}
        self.on_prompt = on_prompt or (lambda screen, question_id: None)
        """Callback when `who`, `candidates`, or `list` starts waiting for keys."""

    def __call__(self, question: Question, index: int, total: int) -> Answer | str:
        while True:
            self._header(question, index, total)
            value = self._select(question)
            if value in (None, QUIT):
                return QUIT
            if value in (SKIP, LIST):
                return value
            if value not in question.pick_for or not question.candidates:
                return Answer(value)
            click.echo()
            click.echo(f"[{index + 1}/{total}] {question.lines[0]}")
            drop = self._drops(question, value)
            if drop is None:
                return QUIT
            if drop != BACK:
                return Answer(value, drop)
            question = replace(question, default=value)

    def _header(self, question: Question, index: int, total: int) -> None:
        click.echo()
        lines = list(question.lines)
        lines[0] = f"[{index + 1}/{total}] {lines[0]}"
        click.echo("\n".join(lines))
        for link in question.links:
            click.echo("  " + link_line(link))

    def pick(self, questions: list[Question], answers: dict[str, Answer]):
        """Shows the questions and returns the selected index or `None` when done."""
        click.echo()
        choices = []
        for index, question in enumerate(questions):
            answer = answers.get(question.id)
            mark = f"✓ {answer.value}" if answer else "· pending"
            choices.append(questionary.Choice(f"{mark:<12} {question.lines[0]}", index))
        choices.append(questionary.Choice("✓ Done", DONE))
        pending = [i for i, q in enumerate(questions) if q.id not in answers]
        prompt = questionary.select(
            f"{len(answers)} of {len(questions)} answered. Ask one again?",
            choices,
            default=choices[pending[0]] if pending else choices[-1],
            **self.io,
        )
        _tune(prompt.application)
        self.on_prompt("list", None)
        value = prompt.ask()
        return None if value in (None, DONE) else value

    def _select(self, question: Question):
        while True:
            choices = self._open_choices(question.links)
            choices += [
                questionary.Choice(f"{o.key}) {o.title}" if o.key else o.title, o.value)
                for o in question.options
            ]
            choices += [
                questionary.Choice("↩ Questions", LIST),
                questionary.Choice("↷ Skip", SKIP),
                questionary.Choice("✕ Quit", QUIT),
            ]
            prompt = questionary.select(
                question.prompt,
                choices,
                default=question.default,
                instruction="(arrows or a roster number · esc questions · q quit)",
                **self.io,
            )
            keys = KeyBindings()
            for option in question.options:
                if option.key:
                    keys.add(option.key)(_exit_with(option.value))
            keys.add("escape", eager=True)(_exit_with(LIST))
            keys.add("q")(_exit_with(QUIT))
            _tune(prompt.application, keys)
            self.on_prompt("who", question.id)
            value = prompt.ask()
            if not isinstance(value, Link):
                return value
            self.open_url(value.url)

    def _open_choices(self, links: list[Link]) -> list:
        if not self.open_url:
            return []
        return [questionary.Choice(open_title(link), link) for link in links]

    def _drops(self, question: Question, value: str):
        """Returns unticked IDs for `value`, `BACK` to go back, or `None` on quit."""
        names = {o.value: o.title for o in question.options}
        if question.earlier and question.earlier.value == value:
            ticked = {c.id for c in question.candidates} - set(question.earlier.drop)
        else:
            # Compilations hold other people's voices too; trust own videos first.
            ticked = {c.id for c in question.candidates if c.owner == value} or {
                c.id for c in question.candidates
            }
        own = [c for c in question.candidates if c.owner == value]
        other = [c for c in question.candidates if c.owner != value]
        choices = []
        for label, group in (
            (f"{names[value]}'s videos", own),
            ("Other videos", other),
        ):
            if group:
                choices.append(questionary.Separator(f"── {label} ──"))
                choices += [
                    questionary.Choice(c.title, c.id, checked=c.id in ticked)
                    for c in group
                ]
        links = {c.id: c.link for c in question.candidates}
        keys = []
        if self.open_url:
            keys.append("o open")
        prompt = questionary.checkbox(
            f"Which candidates are {names[value]}?",
            choices,
            instruction=f"(space tick · {' · '.join(keys + ['enter confirm', 'esc back'])})",
            **self.io,
        )
        app = prompt.application
        control = next(
            c for c in app.layout.find_all_controls() if isinstance(c, InquirerControl)
        )
        bindings = KeyBindings()
        bindings.add("escape", eager=True)(_exit_with(BACK))
        if self.open_url:

            @bindings.add("o")
            def _(event):
                self.open_url(links[control.get_pointed_at().value].url)

        _tune(app, bindings)
        self.on_prompt("candidates", question.id)
        kept = prompt.ask()
        if kept is None or kept == BACK:
            return kept
        return tuple(c.id for c in question.candidates if c.id not in set(kept))


def _tune(app, keys: KeyBindings | None = None) -> None:
    if keys:
        app.key_bindings = merge_key_bindings([app.key_bindings, keys])
    app.ttimeoutlen = ESC_WAIT
    # Without a cursor position reply the prompt only skips reserving rows.
    app.renderer.cpr_not_supported_callback = None


def _exit_with(value):
    def handler(event):
        event.app.exit(result=value)

    return handler
