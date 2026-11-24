from whodat.hints import Profile, rank_profiles


def test_persons_rank_by_mean_standardized_distance():
    turn = {"f0_median": 110.0, "speaking_rate": 3.0}
    profiles = {
        "chris": Profile(
            mean={"f0_median": 100.0, "speaking_rate": 3.0},
            std={"f0_median": 10.0, "speaking_rate": 1.0},
        ),
        "zach": Profile(
            mean={"f0_median": 140.0, "speaking_rate": 2.0},
            std={"f0_median": 10.0, "speaking_rate": 1.0},
        ),
    }

    assert rank_profiles(turn, profiles) == [("chris", 0.5), ("zach", 2.0)]
