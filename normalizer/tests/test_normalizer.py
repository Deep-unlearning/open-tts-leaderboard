"""Parenthetical speech and annotation handling without model dependencies."""

import unittest

from normalizer import BasicMultilingualTextNormalizer, EnglishTextNormalizer
from normalizer.normalizer import remove_non_speech_annotations


class AnnotationNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizers = (
            EnglishTextNormalizer(),
            BasicMultilingualTextNormalizer(),
            BasicMultilingualTextNormalizer(remove_diacritics=False),
        )

    def assert_normalized(self, source, expected):
        for normalizer in self.normalizers:
            with self.subTest(normalizer=type(normalizer).__name__, source=source):
                self.assertEqual(normalizer(source).strip(), expected)

    def test_preserves_spoken_parentheticals_from_eval_data(self):
        self.assert_normalized("The charity (RSPCA) helps animals.", "the charity rspca helps animals")
        self.assert_normalized("Du courrier (ou spam).", "du courrier ou spam")
        source = "A distance of (1,040 kilometers)."
        self.assertEqual(self.normalizers[0](source).strip(), "a distance of 1040 kilometers")
        for normalizer in self.normalizers[1:]:
            self.assertEqual(normalizer(source), "a distance of 1 040 kilometers")

    def test_preserves_ordinary_words_in_brackets(self):
        for opening, closing in (("(", ")"), ("[", "]"), ("<", ">"), ("（", "）"), ("［", "］"), ("＜", "＞")):
            with self.subTest(opening=opening):
                self.assert_normalized(f"before {opening}spoken words{closing} after", "before spoken words after")

    def test_removes_only_exact_non_speech_tags(self):
        for tag in ("laugh", "laughs", "laughing", "laughter", "cough", "coughs", "coughing", "unk"):
            for opening, closing in (("(", ")"), ("[", "]"), ("<", ">"), ("（", "）"), ("［", "］"), ("＜", "＞")):
                with self.subTest(tag=tag, opening=opening):
                    self.assert_normalized(f"before{opening} \t{tag.upper()}\n {closing}after", "before after")

    def test_preserves_longer_phrases_and_unenclosed_tag_words(self):
        self.assert_normalized("Try (cough medicine) and [laughing out loud].", "try cough medicine and laughing out loud")
        self.assert_normalized("laugh cough laughing unknown", "laugh cough laughing unknown")
        self.assert_normalized("before (music) [pause] <speaker> after", "before music pause speaker after")

    def test_handles_nested_tags_without_discarding_spoken_outer_content(self):
        self.assert_normalized("before (spoken [cough] words) after", "before spoken words after")
        self.assert_normalized("before [spoken （LAUGH） words] after", "before spoken words after")
        self.assert_normalized("before (laugh [cough]) after", "before after")
        self.assert_normalized("before (outer (inner words) end) after", "before outer inner words end after")

    def test_requires_matching_complete_delimiters(self):
        for malformed in ("(laugh]", "[cough)", "<unk]", "(laugh", "cough)", "（laugh)", "(cough）"):
            with self.subTest(malformed=malformed):
                word = "unk" if "unk" in malformed else "laugh" if "laugh" in malformed else "cough"
                self.assert_normalized(f"before {malformed} after", f"before {word} after")

    def test_helper_preserves_unknown_enclosures_and_is_idempotent(self):
        source = "(RSPCA) [cough medicine] <ordinary> （spoken words）"
        self.assertEqual(remove_non_speech_annotations(source), source)
        cleaned = remove_non_speech_annotations("before (spoken [cough] words) after (laugh)")
        self.assertEqual(remove_non_speech_annotations(cleaned), cleaned)


if __name__ == "__main__":
    unittest.main()
