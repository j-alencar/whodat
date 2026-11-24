import pytest

from whodat.match import Decision, Thresholds, Turn, decide, identify

THRESHOLDS = Thresholds(high=0.6, low=0.3)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"chris": 0.7, "zach": 0.2}, Decision("chris", 0.7, "chris")),
        ({"chris": 0.25, "zach": 0.1}, Decision("unknown", 0.25, "chris")),
        ({"chris": 0.2, "zach": 0.45}, Decision("review", 0.45, "zach")),
    ],
    ids=["accept-above-high", "reject-below-low", "review-in-between"],
)
def test_top_match_band_picks_identity(scores, expected):
    assert decide(scores, THRESHOLDS) == expected


VOICEPRINTS = {"chris": [1.0, 0.0], "zach": [0.0, 1.0]}


def test_cluster_strategy_names_weak_turn_by_its_cluster():
    turns = [Turn(0.0, 4.0, "A"), Turn(5.0, 6.0, "A")]
    embeddings = [[1.0, 0.0], [0.45, 0.89]]  # alone, the short turn is review

    predictions = identify(turns, turns, embeddings, VOICEPRINTS, THRESHOLDS)

    assert [p.decision.identity for p in predictions] == ["chris", "chris"]


def test_turn_override_renames_turn_confidently_matching_another_person():
    turns = [Turn(0.0, 4.0, "A"), Turn(5.0, 9.0, "A"), Turn(10.0, 12.0, "A")]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]  # diarizer merged zach in

    predictions = identify(
        turns, turns, embeddings, VOICEPRINTS, THRESHOLDS, "cluster+turn-override"
    )

    assert [p.decision.identity for p in predictions] == ["chris", "chris", "zach"]


def test_turn_too_short_to_embed_takes_its_cluster_identity():
    turns = [Turn(0.0, 4.0, "A"), Turn(5.0, 5.3, "A")]
    embeddings = [[1.0, 0.0], [float("nan"), float("nan")]]

    predictions = identify(
        turns, turns, embeddings, VOICEPRINTS, THRESHOLDS, "cluster+turn-override"
    )

    assert [p.decision.identity for p in predictions] == ["chris", "chris"]


def test_crosstalk_keeps_both_voices_named_from_their_clean_speech():
    turns = [Turn(0.0, 6.0, "A"), Turn(4.0, 10.0, "B")]
    exclusive = [Turn(0.0, 5.0, "A"), Turn(5.0, 10.0, "B")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    predictions = identify(turns, exclusive, embeddings, VOICEPRINTS, THRESHOLDS)

    assert [(p.turn, p.decision.identity) for p in predictions] == [
        (Turn(0.0, 6.0, "A"), "chris"),
        (Turn(4.0, 10.0, "B"), "zach"),
    ]


@pytest.mark.parametrize(
    "strategy", ["cluster", "cluster+turn-override"], ids=lambda s: s
)
def test_turn_with_no_clean_speech_takes_its_cluster_identity(strategy):
    turns = [Turn(0.0, 4.0, "A"), Turn(5.0, 9.0, "B"), Turn(6.0, 7.0, "A")]
    exclusive = [Turn(0.0, 4.0, "A"), Turn(5.0, 9.0, "B")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    predictions = identify(
        turns, exclusive, embeddings, VOICEPRINTS, THRESHOLDS, strategy
    )

    assert [p.decision.identity for p in predictions] == ["chris", "zach", "chris"]
