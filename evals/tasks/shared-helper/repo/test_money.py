import unittest

from money import average_price, format_price, order_total


class FormatPriceTests(unittest.TestCase):
    def test_whole_and_part_dollars(self):
        self.assertEqual(format_price(1200), "$12.00")
        self.assertEqual(format_price(1050), "$10.50")
        self.assertEqual(format_price(5), "$0.05")


class TotalsTests(unittest.TestCase):
    def test_order_total_is_a_number(self):
        total = order_total([250, 125])
        self.assertEqual(total, 3.75)
        self.assertGreater(total + 1, 4)

    def test_average_price(self):
        self.assertEqual(average_price([100, 200, 400]), 2.33)
        self.assertEqual(average_price([]), 0.0)


if __name__ == "__main__":
    unittest.main()
