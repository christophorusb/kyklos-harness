import unittest

from pager import page, page_count


class PageTests(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(page(list(range(10)), 1, 3), [0, 1, 2])

    def test_middle_page(self):
        self.assertEqual(page(list(range(10)), 2, 3), [3, 4, 5])

    def test_last_page_is_partial(self):
        self.assertEqual(page(list(range(10)), 4, 3), [9])

    def test_past_the_end_is_empty(self):
        self.assertEqual(page(list(range(10)), 5, 3), [])

    def test_page_numbers_start_at_one(self):
        with self.assertRaises(ValueError):
            page([1, 2], 0, 3)

    def test_page_count(self):
        self.assertEqual(page_count(list(range(10)), 3), 4)


if __name__ == "__main__":
    unittest.main()
