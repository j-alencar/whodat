from whodat.labels import Label, LabelFile, Region
from whodat.match import Decision, Prediction, Turn
from whodat.metrics import ScoreSheet


def predict(start, end, identity, hint=None):
    hint = hint or identity
    return Prediction(Turn(start, end, "A"), Decision(identity, 0.5, hint))


def chris_talks_for_ten_seconds():
    return LabelFile(scored=[Region(0.0, 10.0)], labels=[Label(0.0, 10.0, "chris")])


def test_abstain_rate_is_share_of_review_time():
    sheet = ScoreSheet()

    sheet.add(
        chris_talks_for_ten_seconds(),
        [predict(0.0, 6.0, "chris"), predict(6.0, 10.0, "review", "chris")],
    )

    assert sheet.report()["abstain_rate"] == 0.4


def test_accepted_accuracy_ignores_review_time():
    sheet = ScoreSheet()

    sheet.add(
        chris_talks_for_ten_seconds(),
        [
            predict(0.0, 6.0, "chris"),
            predict(6.0, 8.0, "zach"),
            predict(8.0, 10.0, "review", "zach"),
        ],
    )

    assert sheet.report()["accepted_accuracy"] == 0.75


def test_unsure_labels_are_excluded_from_identification():
    label_file = LabelFile(
        scored=[Region(0.0, 10.0)],
        labels=[Label(0.0, 5.0, "chris"), Label(5.0, 10.0, "?")],
    )
    sheet = ScoreSheet()

    sheet.add(label_file, [predict(0.0, 5.0, "chris"), predict(5.0, 10.0, "zach")])

    assert sheet.report()["accepted_accuracy"] == 1.0


def test_forced_accuracy_takes_top_person_for_review_time():
    sheet = ScoreSheet()

    sheet.add(
        chris_talks_for_ten_seconds(),
        [
            predict(0.0, 6.0, "chris"),
            predict(6.0, 8.0, "review", "chris"),
            predict(8.0, 10.0, "review", "zach"),
        ],
    )

    assert sheet.report()["forced_accuracy"] == 0.8


def test_unknown_false_accept_rate_is_unknown_time_named_as_person():
    label_file = LabelFile(
        scored=[Region(0.0, 10.0)], labels=[Label(0.0, 10.0, "unknown")]
    )
    sheet = ScoreSheet()

    sheet.add(
        label_file,
        [
            predict(0.0, 3.0, "chris"),
            predict(3.0, 8.0, "unknown"),
            predict(8.0, 10.0, "review", "zach"),
        ],
    )

    assert sheet.report()["unknown_false_accept_rate"] == 0.3


def chris_talks_five_seconds_of_ten():
    return LabelFile(scored=[Region(0.0, 10.0)], labels=[Label(0.0, 5.0, "chris")])


def test_false_alarm_rate_ignores_unscored_time():
    sheet = ScoreSheet()

    sheet.add(
        chris_talks_five_seconds_of_ten(),
        [
            predict(0.0, 5.0, "chris"),
            predict(6.0, 8.0, "zach"),
            predict(20.0, 30.0, "zach"),
        ],
    )

    assert sheet.report()["false_alarm_rate"] == 0.4


def test_missed_speech_rate_is_labeled_speech_without_turns():
    sheet = ScoreSheet()

    sheet.add(chris_talks_five_seconds_of_ten(), [predict(0.0, 3.0, "chris")])

    assert sheet.report()["missed_speech_rate"] == 0.4


def test_sheet_collects_match_scores_by_labeled_identity():
    sheet = ScoreSheet()

    sheet.add(chris_talks_five_seconds_of_ten(), [predict(0.0, 10.0, "chris")])

    assert sheet.matches == {(0.5, "chris", "chris"): 5.0}
