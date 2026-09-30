import pytest

from asr_tool.metrics import character_error_rate, edit_distance, normalize_text, word_error_rate


def test_normalize_removes_case_and_punctuation():
    assert normalize_text("  Hello, WORLD!  It's   fine. ") == "hello world its fine"


def test_normalize_splits_hyphenated_words():
    assert normalize_text("speech-to-text") == "speech to text"


@pytest.mark.parametrize("ref, hyp, expected", [
    ("abc", "abc", 0),
    ("abc", "abd", 1),      # substitution
    ("abc", "ab", 1),       # deletion
    ("abc", "abcd", 1),     # insertion
    ("kitten", "sitting", 3),
    ("", "abc", 3),
])
def test_edit_distance(ref, hyp, expected):
    assert edit_distance(ref, hyp) == expected


def test_wer_perfect_match_ignores_formatting():
    assert word_error_rate("The quick brown fox.", "the quick brown fox") == 0.0


def test_wer_counts_substitution_deletion_insertion():
    ref = "the cat sat on the mat"          # 6 words
    hyp = "the cat sit on mat today"        # sat->sit, delete "the", insert "today"
    assert word_error_rate(ref, hyp) == pytest.approx(3 / 6)


def test_wer_can_exceed_one():
    assert word_error_rate("hi", "hello there friend") == pytest.approx(3.0)


def test_wer_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "something") == 1.0


def test_cer():
    assert character_error_rate("speech", "speach") == pytest.approx(1 / 6)
