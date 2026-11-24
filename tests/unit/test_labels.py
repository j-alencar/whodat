from whodat.labels import Label, LabelFile, Region, clock


def test_label_file_reads_scored_regions_and_labels():
    text = "# scored 00:10:00-00:25:00\n00:10:02.4\t00:10:07.9\tchris\n"

    label_file = LabelFile.parse(text)

    assert label_file == LabelFile(
        scored=[Region(600.0, 1500.0)],
        labels=[Label(602.4, 607.9, "chris")],
    )


def test_label_file_writes_readable_timestamps_sorted_by_start():
    label_file = LabelFile(
        scored=[Region(600.0, 1500.0)],
        labels=[Label(3725.25, 3726.0, "unknown"), Label(602.4, 607.9, "chris")],
    )

    assert label_file.format() == (
        "# scored 00:10:00.0-00:25:00.0\n"
        "00:10:02.4\t00:10:07.9\tchris\n"
        "01:02:05.2\t01:02:06.0\tunknown\n"
    )


def test_audacity_round_trip_keeps_scored_regions():
    label_file = LabelFile(
        scored=[Region(600.0, 1500.0)],
        labels=[Label(602.4, 607.9, "chris"), Label(606.8, 609.1, "zach")],
    )

    assert LabelFile.parse_audacity(label_file.format_audacity()) == label_file


def test_impostor_view_relabels_one_person_as_unknown():
    label_file = LabelFile(
        scored=[Region(0.0, 10.0)],
        labels=[Label(0.0, 4.0, "chris"), Label(5.0, 9.0, "zach")],
    )

    assert label_file.without("zach").labels == [
        Label(0.0, 4.0, "chris"),
        Label(5.0, 9.0, "unknown"),
    ]


def test_clock_reads_like_a_youtube_timestamp():
    assert clock(4140.9) == "1:09:00"
    assert clock(65.0) == "1:05"


def test_discarding_labels_matches_them_as_the_file_rounds_them():
    label_file = LabelFile.parse(
        "# scored 00:00:00.0-00:01:00.0\n"
        "00:00:01.2\t00:00:03.0\tchris\n"
        "00:00:01.2\t00:00:03.0\tchris\n"
        "00:00:04.0\t00:00:05.0\tlyle\n"
    )

    label_file.discard([Label(1.234, 2.987, "chris"), Label(4.0, 5.0, "zach")])

    assert label_file.labels == [Label(1.2, 3.0, "chris"), Label(4.0, 5.0, "lyle")]
