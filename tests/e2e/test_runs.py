import json
from pathlib import Path

import e2e_world as world

from whodat.workflows import Workspace


def test_evaluations_and_timelines_go_under_one_runs_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    corpus = world.build(tmp_path, enrolled=True)
    runs = tmp_path / "runs"
    ws = Workspace(corpus, "tone", embedders=world.EMBEDDERS, runs=runs)
    turns = [
        [s, e, cluster]
        for cluster, (_, spans) in world.CLUSTERS.items()
        for s, e in spans
    ]
    cache = world.CACHE / "whodat" / "diarization" / "EVA" / "0.0-30.0.turns.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"turns": turns, "exclusive": turns}))
    labels = [
        f"00:00:{s:04.1f}\t00:00:{e:04.1f}\t{'unknown' if who == 'stranger' else who}"
        for who, spans in world.CLUSTERS.values()
        for s, e in spans
    ]
    (corpus / "labels" / "EVA.tsv").write_text(
        "# scored 00:00:00.0-00:00:30.0\n" + "\n".join(labels) + "\n"
    )

    evaluation = Path(ws.evaluate()["folder"])
    timeline = ws.timeline("EVA")

    assert evaluation.parent == runs / "evaluations"
    assert {p.name for p in evaluation.iterdir()} == {
        "metrics.json",
        "scores.png",
        "confusion.png",
    }
    assert timeline == runs / "timelines" / "EVA.html"
