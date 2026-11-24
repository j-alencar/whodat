import json

import e2e_world as world

from whodat.labels import Region
from whodat.workflows import Workspace


def test_crosstalk_labels_both_voices(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    corpus = world.build(tmp_path, enrolled=True)
    ws = Workspace(corpus, "tone", embedders=world.EMBEDDERS)
    world.write_audio("TLK", 20, [(0, 12, "chris"), (8, 20, "zach")])
    cache = world.CACHE / "whodat" / "diarization" / "TLK" / "0.0-20.0.turns.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "turns": [[0, 12, "0s/A"], [8, 20, "0s/B"]],
                "exclusive": [[0, 10, "0s/A"], [10, 20, "0s/B"]],
            }
        )
    )

    ws.label("TLK", Region(0.0, 20.0))

    labels = ws.corpus.label_file("TLK").labels
    assert [(lb.start, lb.end, lb.identity) for lb in labels] == [
        (0, 12, "chris"),
        (8, 20, "zach"),
    ]
