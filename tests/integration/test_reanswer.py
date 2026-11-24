import json

import pytest

from whodat.labels import Label
from whodat.workflows import Workspace

MANIFEST = """\
videos:
  VID: eval
"""
ROSTER = """\
persons:
  chris:
    name: Chris
    color: "#2a78d6"
  lyle:
    name: Lyle
    color: "#eb6834"
"""
TURNS = [{"start": 1.234, "end": 2.5}, {"start": 4.0, "end": 6.75}]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "manifest.yaml").write_text(MANIFEST)
    (root / "roster.yaml").write_text(ROSTER)
    ws = Workspace(root)
    path = ws.state / "pending" / "VID.json"
    path.parent.mkdir(parents=True)
    entry = {"cluster": "0s/SPEAKER_00", "hint": "", "score": 0.5, "turns": TURNS}
    path.write_text(json.dumps({"0s/SPEAKER_00": entry}))
    return ws


def test_answering_again_replaces_the_earlier_labels(ws):
    ws.answer("VID", "0s/SPEAKER_00", "chris")
    ws.answer("VID", "0s/SPEAKER_00", "lyle")

    assert ws.corpus.label_file("VID").labels == [
        Label(1.2, 2.5, "lyle"),
        Label(4.0, 6.8, "lyle"),
    ]


def test_an_answered_cluster_is_no_longer_pending(ws):
    ws.answer("VID", "0s/SPEAKER_00", "chris")

    assert ws.pending("VID") == {}
