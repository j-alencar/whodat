from whodat.calibrate import fit_thresholds

# (top score, top Person, labeled identity) -> seconds
MATCHES = {
    (0.9, "chris", "chris"): 10.0,
    (0.8, "zach", "zach"): 10.0,
    (0.7, "chris", "zach"): 1.0,
    (0.6, "zach", "zach"): 9.0,
    (0.5, "chris", "unknown"): 10.0,
    (0.5, "zach", "zach"): 2.0,
    (0.4, "zach", "unknown"): 10.0,
    (0.3, "chris", "chris"): 1.0,
    (0.2, "chris", "unknown"): 19.0,
}


def test_high_is_lowest_score_keeping_accepted_error_in_budget():
    assert fit_thresholds(MATCHES, max_error=0.05).high == 0.6


def test_low_is_highest_score_keeping_rejected_error_in_budget():
    assert fit_thresholds(MATCHES, max_error=0.05).low == 0.5
