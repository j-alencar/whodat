from whodat.enroll import group_voices, propose_clips, rank_clips
from whodat.labels import Region
from whodat.match import Turn


def test_proposals_are_dominant_cluster_speech_free_of_other_voices():
    turns = [
        Turn(0.0, 5.0, "A"),
        Turn(10.0, 12.0, "A"),  # too short
        Turn(20.0, 30.0, "A"),
        Turn(25.0, 26.0, "B"),  # crosstalk inside A's turn
        Turn(40.0, 44.0, "B"),
    ]

    proposals = propose_clips(turns, min_duration=3.0, margin=0.25)

    assert proposals == [Region(0.0, 5.0), Region(20.0, 24.75), Region(26.25, 30.0)]


def test_clips_rank_by_similarity_to_their_peers_dropping_outliers():
    clips = [
        Region(0.0, 5.0),
        Region(10.0, 15.0),
        Region(20.0, 25.0),
        Region(30.0, 35.0),
    ]
    rows = [[1.0, 0.0], [0.8, 0.3], [0.0, 1.0], [0.95, 0.05]]  # third: other voice

    ranked = rank_clips(clips, rows, min_similarity=0.6)

    assert [clip for clip, _ in ranked] == [clips[3], clips[0], clips[1]]


def test_voices_group_candidates_that_sound_alike_longest_first():
    rows = [[1.0, 0.0], [0.0, 1.0], [0.95, 0.1], [0.1, 0.9], [0.05, 1.0]]
    seconds = [10.0, 5.0, 10.0, 5.0, 5.0]

    assert group_voices(rows, seconds, min_similarity=0.5) == [[0, 2], [1, 3, 4]]


def test_a_single_candidate_is_its_own_voice():
    assert group_voices([[1.0, 0.0]], [3.0], 0.5) == [[0]]
