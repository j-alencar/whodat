"""Runs corpus workflows for the CLI. It loads large models only when needed."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np

from whodat import hints, media
from whodat.calibrate import fit_thresholds
from whodat.corpus import Clip, Corpus
from whodat.diarize import Diarizer
from whodat.embed import EMBEDDERS, embed_all
from whodat.enroll import (
    DIFFERENT,
    SAME,
    cut_between,
    group_voices,
    kept_merges,
    merges,
    next_merge,
    propose_clips,
    rank_clips,
    suggested_merge,
)
from whodat.evaluate import write_run
from whodat.labeling import Pending, auto_label
from whodat.labels import UNKNOWN, UNSURE, Region, clock
from whodat.match import Thresholds, match_scores
from whodat.metrics import ScoreSheet
from whodat.paths import cache_dir, state_dir
from whodat.pipeline import (
    build_voiceprints,
    load_predictions,
    mean_unit,
    predict,
    save_predictions,
)
from whodat.questionnaire import Answer, Candidate, Link, Option, Question
from whodat.timeline import render

PROVISIONAL = {"ecapa": Thresholds(0.55, 0.25), "wespeaker": Thresholds(0.6, 0.3)}
"""Fallback thresholds for embedders without saved calibration.

`calibrate` fits new thresholds on labeled `calibrate` videos and saves them.
"""
ENROLL_TARGET = 120.0
"""Seconds of confirmed enrollment clips to aim for per person."""
CLIP_MAX = 15.0
LINK_LEAD = 1.0
"""Seconds of lead-in for sample links, when possible, to catch the first word."""
NOBODY = "none"
"""The enrollment answer for a voice outside the roster."""
CUT_OPTIONS = [
    Option(SAME, "Same person", "1"),
    Option(DIFFERENT, "Different people", "2"),
    Option(UNSURE, "Can't tell"),
]


class Workspace:
    def __init__(
        self,
        corpus: Path,
        embedder: str = "wespeaker",
        strategy: str = "cluster",
        embedders=EMBEDDERS,
        runs: Path = Path("runs"),
    ) -> None:
        self.corpus = Corpus(corpus)
        self.embedder_name = embedder
        self.strategy = strategy
        self.embedders = embedders
        self.runs = runs
        """Holds evaluation runs and timelines."""

    @cached_property
    def embedder(self):
        return self.embedders[self.embedder_name]()

    @cached_property
    def diarizer(self) -> Diarizer:
        return Diarizer()

    @property
    def state(self) -> Path:
        return state_dir() / self.corpus.name

    @property
    def thresholds(self) -> Thresholds:
        return self.corpus.thresholds(self.embedder_name) or PROVISIONAL.get(
            self.embedder_name, PROVISIONAL["wespeaker"]
        )

    @cached_property
    def voiceprints(self) -> dict[str, np.ndarray]:
        return build_voiceprints(self.corpus, self.embedder)

    def predict(self, video: str, regions: list[Region], voiceprints=None):
        voiceprints = self.voiceprints if voiceprints is None else voiceprints
        return predict(
            video,
            regions,
            self.diarizer,
            self.embedder,
            voiceprints,
            self.thresholds,
            self.strategy,
        )

    # Links for human ears

    def _link(self, video: str, region: Region, show_video: bool = False) -> Link:
        label = clock(region.start)
        return Link(
            f"{video} {label}" if show_video else label,
            round(region.end - region.start, 1),
            media.youtube_link(video, max(0.0, region.start - LINK_LEAD)),
        )

    def _roster_options(self) -> list[Option]:
        return [
            Option(p.id, p.name, str(i + 1) if i < 9 else None)
            for i, p in enumerate(self.corpus.persons.values())
        ]

    # Enrollment

    def propose(self, person: str, per_video: int = 3, minutes: float = 10.0) -> list:
        """Proposes voices from sampled windows of the person's prominent videos.

        Uses the cluster with the most speech in each window.
        """
        groups = []
        for video in self.corpus.persons[person].prominent:
            length = media.duration(video)
            for k in range(per_video):
                center = length * (k + 0.5) / per_video
                window = Region(
                    round(max(0.0, center - minutes * 30), 1),
                    round(min(length, center + minutes * 30), 1),
                )
                pieces = [
                    Region(t, min(t + CLIP_MAX, r.end))
                    for r in propose_clips(
                        self.diarizer.diarize(video, window).exclusive
                    )
                    for t in np.arange(r.start, r.end, CLIP_MAX)
                    if min(t + CLIP_MAX, r.end) - t >= 3.0
                ]
                if pieces:
                    groups.append({"video": video, "window": window, "clips": pieces})
        rows = [
            mean_unit(
                embed_all(
                    self.embedder,
                    [media.read(g["video"], c.start, c.end) for c in g["clips"]],
                ),
                np.array([c.end - c.start for c in g["clips"]]),
            )
            for g in groups
        ]
        centroid = mean_unit(np.array(rows), np.ones(len(rows)))
        proposals = []
        for index, (group, row) in enumerate(zip(groups, rows)):
            longest = max(group["clips"], key=lambda c: c.end - c.start)
            proposals.append(
                {
                    "id": f"g{index + 1}",
                    "video": group["video"],
                    "window": asdict(group["window"]),
                    "similarity_to_others": round(float(np.dot(row, centroid)), 3),
                    "seconds": round(sum(c.end - c.start for c in group["clips"]), 1),
                    "clips": [asdict(c) for c in group["clips"]],
                    "link": media.youtube_link(group["video"], longest.start),
                }
            )
        path = self.state / "proposals" / f"{person}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proposals, indent=1))
        return proposals

    def _proposals(self, person: str) -> list[dict]:
        path = self.state / "proposals" / f"{person}.json"
        return json.loads(path.read_text()) if path.exists() else []

    def candidates(self) -> dict[str, dict]:
        """Returns all voices by `person/gN` and proposes missing candidates."""
        result = {}
        for person in self.corpus.persons:
            for p in self._proposals(person) or self.propose(person):
                result[f"{person}/{p['id']}"] = p | {"person": person}
        return result

    def _candidate_rows(self, candidates: dict[str, dict]) -> np.ndarray:
        """Computes a duration-weighted unit embedding per candidate voice.

        The cache key uses the full candidate records.
        """
        key = hashlib.sha1(json.dumps(candidates, sort_keys=True).encode())
        path = (
            cache_dir()
            / "candidates"
            / self.corpus.name
            / f"{self.embedder_name}-{key.hexdigest()[:12]}.npy"
        )
        if path.exists():
            return np.load(path)
        rows = []
        for p in candidates.values():
            clips = [Region(c["start"], c["end"]) for c in p["clips"]]
            audio = [media.read(p["video"], c.start, c.end) for c in clips]
            weights = np.array([c.end - c.start for c in clips])
            rows.append(mean_unit(embed_all(self.embedder, audio), weights))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array(rows))
        return np.array(rows)

    # The voice cut

    def voice_cut(self) -> float | None:
        return self.corpus.voice_cut(self.embedder_name)

    def needs_voice_cut(self) -> bool:
        return self.voice_cut() is None and len(self.candidates()) > 1

    def _cut_path(self) -> Path:
        return self.state / f"voice-cut.{self.embedder_name}.json"

    def _cut_answers(self) -> dict[str, str]:
        path = self._cut_path()
        return json.loads(path.read_text()) if path.exists() else {}

    def all_cut_questions(self) -> list[Question]:
        """One "same person?" question per merge, most similar first."""
        candidates = self.candidates()
        ids = list(candidates)
        names = {p.id: p.name for p in self.corpus.persons.values()}
        questions = []
        for merge in merges(self._candidate_rows(candidates)):
            sides = [[ids[i] for i in merge.left], [ids[i] for i in merge.right]]
            lines, links = [], []
            for tag, members in zip("AB", sides):
                owners = Counter(candidates[m]["person"] for m in members)
                spread = ", ".join(f"{names[p]} ×{n}" for p, n in owners.most_common())
                seconds = sum(candidates[m]["seconds"] for m in members)
                lines.append(
                    f"{tag}  {len(members)} candidates ({spread}) · {seconds:.0f} s"
                )
                longest = max(members, key=lambda m: candidates[m]["seconds"])
                link = self._sample(candidates[longest], show_video=True)
                links.append(replace(link, label=f"{tag} {link.label}"))
            a, b = sorted(min(members) for members in sides)
            questions.append(
                Question(
                    f"{a}+{b}",
                    lines=[f"{a} + {b} · similarity {merge.similarity:.2f}", *lines],
                    links=links,
                    options=CUT_OPTIONS,
                    prompt="Same person?",
                    details={
                        "similarity": merge.similarity,
                        "left": sides[0],
                        "right": sides[1],
                    },
                )
            )
        return questions

    def _settle_cut(self) -> tuple[list[Question], int | None, float]:
        """The cut questions, the one to ask next, and the cut so far."""
        questions = self.all_cut_questions()
        similarities = [q.details["similarity"] for q in questions]
        stored = self._cut_answers()
        answers = {i: stored[q.id] for i, q in enumerate(questions) if q.id in stored}
        asked = next_merge(len(questions), answers, suggested_merge(similarities))
        cut = cut_between(similarities, kept_merges(len(questions), answers))
        return questions, asked, cut

    def cut_questions(self, answers) -> list[Question]:
        """The cut questions answered this session, then the one to ask next."""
        questions, asked, _ = self._settle_cut()
        return [q for i, q in enumerate(questions) if q.id in answers or i == asked]

    def answer_cut(self, question: Question, answer: Answer) -> None:
        """Saves the voice cut once the answers settle it, and clears it until then."""
        stored = self._cut_answers() | {question.id: answer.value}
        path = self._cut_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, indent=1))
        _, asked, cut = self._settle_cut()
        self.corpus.save_voice_cut(self.embedder_name, cut if asked is None else None)

    def reset_voice_cut(self) -> None:
        """Forgets the voice cut, so `enroll review` asks for it again."""
        self._cut_path().unlink(missing_ok=True)
        self.corpus.save_voice_cut(self.embedder_name, None)

    # Voices

    def voices(self) -> list[dict]:
        candidates = self.candidates()
        ids = list(candidates)
        if self.needs_voice_cut():
            raise ValueError("no voice cut: answer the questions of `enroll review`")
        rows = self._candidate_rows(candidates)
        seconds = [candidates[c]["seconds"] for c in ids]
        result = []
        for group in group_voices(rows, seconds, self.voice_cut() or 1.0):
            centroid = mean_unit(rows[group], np.ones(len(group)))
            members = [ids[i] for i in group]
            owners = Counter(candidates[m]["person"] for m in members)
            result.append(
                {
                    "id": members[0],
                    "members": members,
                    "likely": owners.most_common(1)[0][0],
                    "seconds": round(sum(seconds[i] for i in group), 1),
                    "similarity": {
                        ids[i]: round(float(np.dot(rows[i], centroid)), 3)
                        for i in group
                    },
                }
            )
        return result

    def _answered_voices(self) -> dict:
        path = self.state / "voices.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _save_answered_voices(self, answered: dict) -> None:
        path = self.state / "voices.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answered, indent=1))

    def _forget_changed_voices(self, voices: list[dict]) -> None:
        """Clears answers, and the clips they added, of voices that regrouped."""
        current = {frozenset(v["members"]) for v in voices}
        answered = self._answered_voices()
        for vid, entry in list(answered.items()):
            if frozenset(entry["members"]) in current:
                continue
            if entry.get("value") in self.corpus.persons:
                self.corpus.remove_clips(
                    entry["value"], [Clip.parse(c) for c in entry.get("clips", [])]
                )
            del answered[vid]
        self._save_answered_voices(answered)

    def _enrolled(self, proposal: dict) -> bool:
        window = Region(**proposal["window"])
        return any(
            c.video == proposal["video"] and window.start <= c.start < window.end
            for p in self.corpus.persons.values()
            for c in p.clips
        )

    def enrollment_questions(self) -> list[Question]:
        candidates = self.candidates()
        voices = self.voices()
        self._forget_changed_voices(voices)
        answered = [set(v["members"]) for v in self._answered_voices().values()]
        names = {p.id: p.name for p in self.corpus.persons.values()}
        questions = []
        for voice in voices:
            members = voice["members"]
            if set(members) in answered or any(
                self._enrolled(candidates[m]) for m in members
            ):
                continue
            owners = Counter(candidates[m]["person"] for m in members)
            spread = ", ".join(f"{names[p]} ×{n}" for p, n in owners.most_common())
            by_similarity = sorted(members, key=lambda m: -voice["similarity"][m])
            header = (
                f"Voice · {len(members)} candidates ({spread}) · "
                f"{voice['seconds']:.0f} s · likely {names[voice['likely']]}"
            )
            questions.append(
                Question(
                    voice["id"],
                    lines=[header],
                    links=[
                        self._sample(candidates[m], show_video=True)
                        for m in by_similarity[:2]
                    ],
                    options=self._roster_options()
                    + [Option(NOBODY, "Someone not in the roster")],
                    default=voice["likely"],
                    prompt="Who is this Voice?",
                    candidates=[
                        Candidate(
                            m,
                            f"{m:<9} {_window(candidates[m])}  "
                            f"{candidates[m]['seconds']:>4.0f} s  "
                            f"{voice['similarity'][m]:.2f}",
                            self._sample(candidates[m], show_video=True),
                            owner=candidates[m]["person"],
                        )
                        for m in members
                    ],
                    pick_for=frozenset(self.corpus.persons),
                    details=voice,
                )
            )
        return questions

    def _sample(self, proposal: dict, show_video: bool = False) -> Link:
        longest = max(proposal["clips"], key=lambda c: c["end"] - c["start"])
        return self._link(proposal["video"], Region(**longest), show_video)

    def answer_voice(self, question: Question, answer: Answer) -> None:
        """Enrolls clips from kept candidates for the chosen person.

        A new answer replaces the voice's earlier answer.
        """
        answered = self._answered_voices()
        earlier = answered.get(question.id, {})
        if earlier.get("value") in self.corpus.persons:
            self.corpus.remove_clips(
                earlier["value"], [Clip.parse(c) for c in earlier.get("clips", [])]
            )
        members = [m for m in question.details["members"] if m not in answer.drop]
        added = []
        if answer.value != NOBODY and members:
            candidates = self.candidates()
            added = self._enroll(answer.value, [candidates[m] for m in members])
        answered[question.id] = {
            "members": question.details["members"],
            "value": answer.value,
            "drop": list(answer.drop),
            "clips": [c.format() for c in added],
        }
        self._save_answered_voices(answered)

    def reset_enrollment(self, persons: list[str]) -> None:
        """Clears enrollment clips and related answers for `persons`, or for everyone.

        An answer is related if it names a selected person or includes one
        of their candidates. Proposals stay, so `enroll review` then asks
        about their voices again.
        """
        if unknown := sorted(set(persons) - set(self.corpus.persons)):
            raise ValueError(f"not in the Roster: {', '.join(unknown)}")
        persons = persons or list(self.corpus.persons)
        for person in persons:
            self.corpus.clear_clips(person)
        answered = {
            vid: voice
            for vid, voice in self._answered_voices().items()
            if voice["value"] not in persons
            and not any(m.split("/")[0] in persons for m in voice["members"])
        }
        path = self.state / "voices.json"
        if path.exists():
            path.write_text(json.dumps(answered, indent=1))

    def _enroll(
        self, person: str, chosen: list[dict], limit: float = ENROLL_TARGET
    ) -> list[Clip]:
        """Adds the cleanest clips from `chosen` candidate voices within budget."""
        candidates = [
            (index, p["video"], Region(c["start"], c["end"]))
            for index, p in enumerate(chosen)
            for c in p["clips"]
        ]
        rows = embed_all(
            self.embedder, [media.read(v, r.start, r.end) for _, v, r in candidates]
        )
        ranked = rank_clips(candidates, rows)
        have = sum(c.end - c.start for c in self.corpus.persons[person].clips)
        # Spread the budget across chosen candidates, cleanest clips first.
        budget = max(0.0, limit - have) / max(1, len(chosen))
        spent = [0.0] * len(chosen)
        added = []
        for (index, video, region), _ in ranked:
            if spent[index] + region.end - region.start > budget:
                continue
            added.append(Clip(video, region.start, region.end))
            self.corpus.add_clip(person, added[-1])
            spent[index] += region.end - region.start
        return added

    def enrollment_status(self) -> dict:
        return {
            p.id: {
                "clips": len(p.clips),
                "seconds": round(sum(c.end - c.start for c in p.clips), 1),
            }
            for p in self.corpus.persons.values()
        }

    # Semi-auto labeling

    def _pending_path(self, video: str) -> Path:
        return self.state / "pending" / f"{video}.json"

    def _entries(self, video: str) -> dict[str, dict]:
        """Returns keyed review clusters with an `answer` for each answered one."""
        path = self._pending_path(video)
        return json.loads(path.read_text()) if path.exists() else {}

    def _save_entries(self, video: str, entries: dict[str, dict]) -> None:
        path = self._pending_path(video)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries))

    def pending(self, video: str) -> dict[str, Pending]:
        return {
            key: _pending(e)
            for key, e in self._entries(video).items()
            if "answer" not in e
        }

    def label(self, video: str, region: Region) -> dict:
        """Adds a scored region, writes automatic labels, and queues `review` clusters.

        For `calibrate` and `eval` videos, every cluster goes to review.
        """
        label_file = self.corpus.label_file(video)
        predictions = self.predict(video, [region])
        held_out = self.corpus.roles.get(video) in ("calibrate", "eval")
        labels, new_pending = auto_label(region, predictions, accept=not held_out)
        label_file.scored.append(region)
        label_file.labels += labels
        self.corpus.save_label_file(video, label_file)
        entries = self._entries(video) | {k: asdict(v) for k, v in new_pending.items()}
        self._save_entries(video, entries)
        return {"auto_labels": len(labels), "review_clusters": len(new_pending)}

    @cached_property
    def profiles(self) -> dict[str, hints.Profile]:
        result = {}
        for person in self.corpus.persons.values():
            samples = [
                hints.features(
                    media.read(c.video, c.start, c.end),
                    _word_count(c.video, c.start, c.end),
                )
                for c in person.clips
            ]
            if samples:
                result[person.id] = hints.build_profile(samples)
        return result

    def review_questions(self, video: str) -> list[Question]:
        names = {p.id: p.name for p in self.corpus.persons.values()}
        high = self.thresholds.high
        questions = []
        pending = sorted(
            self.pending(video).items(),
            key=lambda kv: min(t.start for t in kv[1].turns),
        )
        for key, entry in pending:
            longest = sorted(entry.turns, key=lambda t: t.start - t.end)[:3]
            audio = [media.read(video, t.start, t.end) for t in longest]
            rows = embed_all(self.embedder, audio)
            scores = match_scores(
                mean_unit(rows, np.array([t.end - t.start for t in longest])),
                self.voiceprints,
            )
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            words = sum(_word_count(video, t.start, t.end) for t in longest)
            turn_features = hints.features(np.concatenate(audio), words)
            closest = hints.rank_profiles(turn_features, self.profiles)
            seconds = round(sum(t.end - t.start for t in entry.turns), 1)
            lines = [
                f"Review · {video} · {key} · {seconds} s over {len(entry.turns)} turns"
            ]
            guess = ranked[0][0] if ranked else None
            if ranked:
                person, score = ranked[0]
                line = (
                    f"Guess  {names[person]} {score:.2f}{' ⚠' if score < high else ''}"
                )
                if len(ranked) > 1:
                    line += f"   runner-up {names[ranked[1][0]]} {ranked[1][1]:.2f}"
                lines.append(line)
            if closest:
                lines.append(
                    "Hints  closest acoustic profiles: "
                    + ", ".join(names[p] for p, _ in closest[:2])
                )
            questions.append(
                Question(
                    key,
                    lines=lines,
                    links=[self._link(video, t) for t in longest[:2]],
                    options=self._roster_options()
                    + [
                        Option(UNKNOWN, "Someone not in the roster"),
                        Option(UNSURE, "Can't tell"),
                    ],
                    default=guess,
                    details={
                        "turns": len(entry.turns),
                        "seconds": seconds,
                        "scores": {p: round(s, 3) for p, s in ranked},
                        "hints": {
                            "turn": {k: round(v, 1) for k, v in turn_features.items()},
                            "closest_profiles": closest,
                        },
                    },
                )
            )
        return questions

    def answer(self, video: str, cluster: str, identity: str) -> int:
        """Labels the cluster's turns as `identity`, replacing any earlier answer."""
        entries = self._entries(video)
        entry = _pending(entries[cluster])
        label_file = self.corpus.label_file(video)
        if earlier := entries[cluster].get("answer"):
            label_file.discard(entry.answer(earlier))
        label_file.labels += entry.answer(identity)
        self.corpus.save_label_file(video, label_file)
        entries[cluster]["answer"] = identity
        self._save_entries(video, entries)
        return len(entry.turns)

    # Calibration and Evaluation Runs

    def _score(self, videos: list[str]) -> tuple[dict, ScoreSheet, ScoreSheet]:
        """Returns per-video, total, and impostor score sheets.

        For each enrolled person, impostor scoring removes their voiceprint
        and keeps only their speech and speech already labeled `unknown` in
        the reference, with both labeled `unknown`.
        """
        sheets, total, impostor = {}, ScoreSheet(), ScoreSheet()
        for video in videos:
            label_file = self.corpus.label_file(video)
            if not label_file.scored:
                continue
            predictions = self.predict(video, label_file.scored)
            sheets[video] = ScoreSheet()
            sheets[video].add(label_file, predictions)
            total.add(label_file, predictions)
            for person in self.voiceprints:
                others = {p: v for p, v in self.voiceprints.items() if p != person}
                view = label_file.without(person)
                view.labels = [lb for lb in view.labels if lb.identity == UNKNOWN]
                impostor.add(view, self.predict(video, label_file.scored, others))
        return sheets, total, impostor

    def calibrate(self, max_error: float = 0.05) -> Thresholds:
        _, total, impostor = self._score(self.corpus.videos("calibrate"))
        if not total.matches:
            raise ValueError("no labeled Scored Regions on calibrate Videos")
        thresholds = fit_thresholds(
            Counter(total.matches) + Counter(impostor.matches), max_error
        )
        self.corpus.save_thresholds(self.embedder_name, thresholds)
        return thresholds

    def evaluate(self, out: Path | None = None) -> dict:
        sheets, total, impostor = self._score(self.corpus.videos("eval"))
        if not sheets:
            raise ValueError("no labeled Scored Regions on eval Videos")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = (out or self.runs / "evaluations") / (
            f"{stamp}-{self.embedder_name}-{self.strategy}"
        )
        settings = {
            "corpus": self.corpus.name,
            "embedder": self.embedder_name,
            "strategy": self.strategy,
            "thresholds_calibrated": self.corpus.thresholds(self.embedder_name)
            is not None,
        }
        result = write_run(
            folder,
            sheets,
            total,
            impostor,
            self.thresholds,
            list(self.corpus.persons),
            settings,
        )
        return {"folder": str(folder), **result}

    # Timeline

    def timeline(
        self, video: str, out: Path | None = None, title: str | None = None
    ) -> Path:
        out = out or self.runs / "timelines" / f"{video}.html"
        length = media.duration(video)
        t = self.thresholds
        prints = hashlib.sha1()
        for person, print_ in sorted(self.voiceprints.items()):
            prints.update(person.encode() + np.asarray(print_).tobytes())
        cache = (
            cache_dir()
            / "predictions"
            / self.corpus.name
            / f"{video}-{self.embedder_name}-{self.strategy}-{t.high}-{t.low}"
            f"-{prints.hexdigest()[:12]}.json"
        )
        if cache.exists():
            predictions = load_predictions(cache)
        else:
            predictions = self.predict(video, [Region(0.0, length)])
            save_predictions(cache, predictions)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render(
                video,
                title or video,
                length,
                list(self.corpus.persons.values()),
                predictions,
            )
        )
        return out


def _pending(entry: dict) -> Pending:
    turns = [Region(**t) for t in entry["turns"]]
    return Pending(entry["cluster"], entry["hint"], entry["score"], turns)


def _window(proposal: dict) -> str:
    """Formats the video ID and sampled window start, as in `DZv_IFWdUeg   13:44`."""
    window = Region(**proposal["window"])
    return f"{proposal['video']} {clock(window.start):>7}"


def _word_count(video: str, start: float, end: float) -> int:
    return sum(1 for t, _ in media.words(video) if start <= t < end)
