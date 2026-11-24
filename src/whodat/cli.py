"""Runs whodat's CLI. Interactive reviews print text and data commands print JSON."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from whodat import media
from whodat.embed import EMBEDDERS
from whodat.labels import LabelFile, Region, parse_time
from whodat.match import STRATEGIES
from whodat.questionnaire import (
    Answer,
    Apply,
    Question,
    Questions,
    dump_answers,
    from_answers,
    parse_answers,
    parse_sets,
    run,
)
from whodat.tui import Tui, browser
from whodat.workflows import Workspace


class RegionType(click.ParamType):
    name = "START-END"

    def convert(self, value, param, ctx) -> Region:
        try:
            start, end = value.split("-")
            return Region(parse_time(start), parse_time(end))
        except ValueError:
            self.fail(f"{value!r} is not HH:MM:SS-HH:MM:SS", param, ctx)


def emit(value) -> None:
    click.echo(json.dumps(value, indent=1, default=_jsonable))


def _jsonable(value):
    return sorted(value) if isinstance(value, (set, frozenset)) else str(value)


def answer_options(command):
    """Adds options to list pending questions or answer them without the TUI."""
    for option in reversed(
        [
            click.option(
                "-a",
                "--answers",
                type=click.Path(exists=True, dir_okay=False, path_type=Path),
                help="Answers file: YAML of ID: VALUE.",
            ),
            click.option(
                "-s",
                "--set",
                "sets",
                multiple=True,
                metavar="ID=VALUE",
                help="Answer one Question. Repeatable.",
            ),
            click.option(
                "-l", "--list", "list_", is_flag=True, help="Print pending Questions."
            ),
        ]
    ):
        command = option(command)
    return command


def given_answers(
    answers: Path | None, sets: tuple[str, ...], drops: tuple[str, ...] = ()
) -> dict[str, Answer] | None:
    """Answers given as options, or None to ask in the TUI."""
    if not (answers or sets or drops):
        return None
    try:
        given = parse_answers(answers.read_text()) if answers else {}
        return given | parse_sets(list(sets))
    except ValueError as error:
        raise click.UsageError(str(error)) from error


def questionnaire(
    ws: Workspace,
    name: str,
    questions: Questions,
    apply: Apply,
    given: dict[str, Answer] | None,
    list_: bool,
    drops: tuple[str, ...] = (),
    empty: str = "No pending questions.",
    tui: Tui | None = None,
    known: list[Question] | None = None,
):
    """Lists `questions`, answers them from `given`, or asks them in the TUI.

    Returns what to print as JSON, or None after the TUI. `known` holds every
    question `given` may answer, when `questions` changes with the answers.
    """
    current = questions if callable(questions) else lambda answers: questions
    if list_:
        return [asdict(q) for q in current({})]
    if given is not None:
        try:
            ask = from_answers(given, known or current({}), drops)
        except ValueError as error:
            raise click.UsageError(str(error)) from error
        return {q.id: asdict(a) for q, a in run(questions, ask, apply)}
    if not current({}):
        click.echo(empty)
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = ws.state / "answers" / f"{name}-{stamp}.yaml"
    recorded: dict[str, Answer] = {}

    def record(question: Question, answer: Answer) -> None:
        recorded[question.id] = answer
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_answers(recorded))

    tui = tui or Tui(browser())
    done = run(questions, tui, apply, record, pick=tui.pick)
    click.echo()
    for question, answer in done:
        dropped = f"  dropped {', '.join(answer.drop)}" if answer.drop else ""
        click.echo(f"{question.id:<24} {answer.value}{dropped}")
    click.echo(f"{len(done)} answered.")
    if recorded:
        click.echo(f"Answers: {path}")
    return None


pass_ws = click.make_pass_decorator(Workspace)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--corpus",
    envvar="WHODAT_CORPUS",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Corpus directory. Env: WHODAT_CORPUS.",
)
@click.option(
    "-e",
    "--embedder",
    type=click.Choice(sorted(EMBEDDERS)),
    default="wespeaker",
    show_default=True,
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(STRATEGIES),
    default="cluster",
    show_default=True,
    help="Match Strategy.",
)
@click.pass_context
def cli(ctx, corpus: Path, embedder: str, strategy: str) -> None:
    """Identifies who spoke when in YouTube videos."""
    ctx.obj = Workspace(corpus, embedder, strategy)


@cli.command()
@pass_ws
def fetch(ws: Workspace) -> None:
    """Downloads and converts audio for every video in the manifest."""
    emit({v: round(media.duration(v)) for v in ws.corpus.roles})


@cli.group()
def enroll() -> None:
    """Builds voiceprints from videos where roster members are prominent."""


@enroll.command("propose")
@click.argument("person")
@click.option(
    "-n",
    "--per-video",
    default=3,
    show_default=True,
    help="Windows sampled per Prominent Video.",
)
@click.option("-m", "--minutes", default=10.0, show_default=True, help="Window length.")
@pass_ws
def enroll_propose(ws: Workspace, person: str, per_video: int, minutes: float) -> None:
    """Proposes candidate voices for the specified person."""
    proposals = ws.propose(person, per_video, minutes)
    emit([{k: v for k, v in p.items() if k != "clips"} for p in proposals])


@enroll.command("review")
@answer_options
@click.option(
    "-d",
    "--drop",
    "drops",
    multiple=True,
    metavar="CANDIDATE",
    help="Leave out a Candidate Voice, e.g. tomar/g3. Repeatable.",
)
@pass_ws
def enroll_review(ws: Workspace, answers, sets, list_, drops) -> None:
    """Asks who each voice belongs to and enrolls clips from the candidates you keep.

    First, while no voice cut is saved for the embedder, asks whether pairs of
    candidate groups are the same person. The answers set the cut that groups
    candidates into voices.
    """
    given = given_answers(answers, sets, drops)
    printed = {}
    if ws.needs_voice_cut():
        known = ws.all_cut_questions()
        ids = {q.id for q in known}
        result = questionnaire(
            ws,
            "voice-cut",
            ws.cut_questions,
            ws.answer_cut,
            None if given is None else {k: a for k, a in given.items() if k in ids},
            list_,
            known=known,
        )
        if ws.needs_voice_cut():
            if result is None:
                click.echo("The voice cut is not settled. Run this again to go on.")
            else:
                emit(result)
                if any(k not in ids for k in given or {}):
                    click.echo(
                        "Voice answers wait until the voice cut settles.", err=True
                    )
            return
        if result is None:
            click.echo(f"Voice cut for {ws.embedder_name}: {ws.voice_cut()}")
        else:
            printed = result
            given = {k: a for k, a in given.items() if k not in ids}
    result = questionnaire(
        ws,
        "enroll",
        ws.enrollment_questions(),
        ws.answer_voice,
        given,
        list_,
        drops,
        empty="No pending questions: every Voice has Enrollment Clips or an answer.\n"
        "To enroll again, run `whd enroll reset [PERSON...]`.",
    )
    if result is not None:
        emit(printed | result if isinstance(result, dict) else result)


@enroll.command("reset")
@click.argument("persons", nargs=-1)
@click.option(
    "--voices",
    is_flag=True,
    help="Forget only the voice cut, so `enroll review` asks for it again.",
)
@pass_ws
def enroll_reset(ws: Workspace, persons: tuple[str, ...], voices: bool) -> None:
    """Clears the enrollment clips and answers of PERSONS, or of everyone.

    `enroll review` then asks about their voices again.
    Git tracks `roster.yaml`, so `git restore` undoes this.
    """
    if voices:
        if persons:
            raise click.UsageError("--voices takes no PERSONS")
        ws.reset_voice_cut()
        emit({"embedder": ws.embedder_name, "voice_cut": None})
        return
    try:
        ws.reset_enrollment(list(persons))
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    emit(ws.enrollment_status())


@enroll.command("status")
@pass_ws
def enroll_status(ws: Workspace) -> None:
    """Reports each person's enrollment clip count and total seconds."""
    emit(ws.enrollment_status())


@cli.command()
@click.argument("video")
@click.argument("region", type=RegionType())
@pass_ws
def label(ws: Workspace, video: str, region: Region) -> None:
    """Adds `START-END` as a scored region in `VIDEO` and labels it semi-automatically.

    All clusters in `calibrate` and `eval` videos go to human review.
    """
    emit(ws.label(video, region))


@cli.command()
@click.argument("video")
@answer_options
@pass_ws
def review(ws: Workspace, video: str, answers, sets, list_) -> None:
    """Asks who is speaking in each cluster queued for review in `VIDEO`."""

    def apply(question: Question, answer: Answer) -> None:
        ws.answer(video, question.id, answer.value)

    result = questionnaire(
        ws,
        f"review-{video}",
        ws.review_questions(video),
        apply,
        given_answers(answers, sets),
        list_,
        empty=f"No pending questions: `whd label {video} START-END` queues some.",
    )
    if result is not None:
        emit(result)


@cli.command()
@click.option(
    "--max-error",
    default=0.05,
    show_default=True,
    help="Allowed error share in each auto band.",
)
@pass_ws
def calibrate(ws: Workspace, max_error: float) -> None:
    """Fits thresholds on the labeled `calibrate` videos."""
    thresholds = ws.calibrate(max_error)
    emit({"embedder": ws.embedder_name, "high": thresholds.high, "low": thresholds.low})


@cli.command()
@click.option(
    "-o",
    "--out",
    default="runs",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@pass_ws
def evaluate(ws: Workspace, out: Path) -> None:
    """Evaluates the labeled `eval` videos and writes `metrics.json` and plots."""
    emit(ws.evaluate(out))


@cli.command()
@click.argument("video")
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="HTML file. Default: runs/timelines/VIDEO.html",
)
@click.option("-t", "--title", help="Page title. Default: VIDEO.")
@pass_ws
def timeline(ws: Workspace, video: str, out: Path | None, title: str | None) -> None:
    """Renders the timeline for `VIDEO` as interactive HTML."""
    out = out or Path("runs") / "timelines" / f"{video}.html"
    emit({"timeline": str(ws.timeline(video, out, title))})


@cli.command()
@click.argument("video")
@click.option("-l", "--load", is_flag=True, help="Read the Audacity file back.")
@pass_ws
def audacity(ws: Workspace, video: str, load: bool) -> None:
    """Exports the label file for `VIDEO` to Audacity or imports it with `--load`."""
    path = ws.corpus.label_path(video).with_suffix(".audacity.txt")
    if load:
        ws.corpus.save_label_file(video, LabelFile.parse_audacity(path.read_text()))
    else:
        path.write_text(ws.corpus.label_file(video).format_audacity())
    emit({"audacity": str(path), "loaded": load})
