from whodat.labeling import auto_label
from whodat.labels import Label, Region
from whodat.match import Decision, Prediction, Turn


def test_accepted_turns_become_labels_clipped_to_region():
    predictions = [
        Prediction(Turn(590.0, 605.0, "A"), Decision("chris", 0.8, "chris")),
        Prediction(Turn(606.0, 609.0, "B"), Decision("review", 0.5, "zach")),
        Prediction(Turn(610.0, 612.0, "C"), Decision("unknown", 0.1, "zach")),
    ]

    labels, _ = auto_label(Region(600.0, 900.0), predictions)

    assert labels == [Label(600.0, 605.0, "chris"), Label(610.0, 612.0, "unknown")]


def test_answering_review_cluster_labels_all_its_turns():
    predictions = [
        Prediction(Turn(606.0, 609.0, "B"), Decision("review", 0.5, "zach")),
        Prediction(Turn(700.0, 703.0, "B"), Decision("review", 0.5, "zach")),
    ]
    _, pending = auto_label(Region(600.0, 900.0), predictions)

    assert pending["B"].answer("lyle") == [
        Label(606.0, 609.0, "lyle"),
        Label(700.0, 703.0, "lyle"),
    ]


def test_without_auto_accept_every_cluster_awaits_a_human():
    predictions = [
        Prediction(Turn(606.0, 609.0, "A"), Decision("chris", 0.8, "chris")),
        Prediction(Turn(610.0, 612.0, "B"), Decision("unknown", 0.1, "zach")),
    ]

    labels, pending = auto_label(Region(600.0, 900.0), predictions, accept=False)

    assert (labels, sorted(pending)) == ([], ["A", "B"])
