from whodat.corpus import Person
from whodat.match import Decision, Prediction, Turn
from whodat.timeline import page_data

PERSONS = [Person("chris", "Chris", "#2a78d6"), Person("zach", "Zach", "#eda100")]


def said(start, end, identity):
    return Prediction(Turn(start, end, "A"), Decision(identity, 0.5, identity))


def test_totals_sum_speaking_time_per_identity_in_row_order():
    predictions = [
        said(0, 2, "unknown"),
        said(2, 5, "zach"),
        said(5, 6, "zach"),
        said(6, 10, "chris"),
    ]

    data = page_data("vid", 10.0, PERSONS, predictions)

    assert data["totals"] == [["chris", 4.0, 1], ["zach", 4.0, 2], ["unknown", 2.0, 1]]
