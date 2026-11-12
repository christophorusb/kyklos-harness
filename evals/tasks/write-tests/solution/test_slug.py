import unittest

from slug import slugify


class SlugifyTests(unittest.TestCase):
    def test_accents_are_dropped(self):
        self.assertEqual(slugify("Crème Brûlée"), "creme-brulee")

    def test_punctuation_separates_words(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_runs_of_separators_collapse(self):
        self.assertEqual(slugify("a -- b __ c"), "a-b-c")

    def test_leading_and_trailing_spaces(self):
        self.assertEqual(slugify("   padded title   "), "padded-title")

    def test_custom_sep(self):
        self.assertEqual(slugify("Two Words", sep="_"), "two_words")


if __name__ == "__main__":
    unittest.main()
