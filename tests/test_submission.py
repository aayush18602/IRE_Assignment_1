from ire_a1.submission import format_submission_line, ranks_from_scores


def test_ranks_from_scores_highest_score_gets_rank_1():
    ranks = ranks_from_scores([0.1, 0.9, 0.5])
    assert ranks == [3, 1, 2]


def test_ranks_from_scores_is_a_valid_permutation():
    scores = [0.3, 0.1, 0.9, 0.5, 0.2]
    ranks = ranks_from_scores(scores)
    assert sorted(ranks) == list(range(1, len(scores) + 1))


def test_ranks_from_scores_ties_broken_by_original_order():
    ranks = ranks_from_scores([0.0, 0.0, 0.0])
    assert ranks == [1, 2, 3]


def test_format_submission_line_matches_official_example():
    # from the PDF's own worked example
    assert format_submission_line(6451339, [8, 1, 6, 7, 4, 2, 9, 5, 3]) == "6451339 [8,1,6,7,4,2,9,5,3]\n"
