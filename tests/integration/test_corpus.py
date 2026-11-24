import pytest

from whodat.corpus import Clip, Corpus
from whodat.match import Thresholds

MANIFEST = """\
videos:
  -VwAMsgAUvA: eval
  DZv_IFWdUeg: enroll
"""
ROSTER = """\
persons:
  chris:
    name: Chris
    color: "#e4572e"
    prominent: [DZv_IFWdUeg]
"""


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "manifest.yaml").write_text(MANIFEST)
    (tmp_path / "roster.yaml").write_text(ROSTER)
    return Corpus(tmp_path)


def test_confirmed_enrollment_clip_persists_in_roster(corpus):
    clip = Clip("DZv_IFWdUeg", 63.0, 75.5)

    corpus.add_clip("chris", clip)

    assert Corpus(corpus.root).persons["chris"].clips == [clip]


def test_enrollment_clip_from_held_out_video_is_rejected(corpus):
    with pytest.raises(ValueError, match="not an enroll Video"):
        corpus.add_clip("chris", Clip("-VwAMsgAUvA", 10.0, 20.0))


def test_clearing_clips_empties_a_persons_enrollment(corpus):
    corpus.add_clip("chris", Clip("DZv_IFWdUeg", 63.0, 75.5))

    corpus.clear_clips("chris")

    assert Corpus(corpus.root).persons["chris"].clips == []


def test_calibrating_keeps_the_voice_cut(corpus):
    corpus.save_voice_cut("wespeaker", 0.65)

    corpus.save_thresholds("wespeaker", Thresholds(0.6, 0.3))

    assert corpus.voice_cut("wespeaker") == 0.65
    assert corpus.thresholds("wespeaker") == Thresholds(0.6, 0.3)


def test_a_voice_cut_alone_leaves_the_thresholds_unset(corpus):
    corpus.save_voice_cut("ecapa", 0.7)

    assert corpus.thresholds("ecapa") is None
