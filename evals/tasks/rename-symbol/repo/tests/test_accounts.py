import unittest

from accounts import admin, api, store


class AccountTests(unittest.TestCase):
    def test_lookup(self):
        self.assertEqual(store.get_usr(1)["name"], "Ada")

    def test_missing_user(self):
        self.assertIsNone(store.get_usr(99))

    def test_email(self):
        self.assertEqual(store.get_usr_email(2), "lin@example.com")

    def test_api(self):
        self.assertEqual(api.show_user(2)["email"], "lin@example.com")

    def test_admin(self):
        self.assertEqual(admin.describe(3), "#3 (missing)")


if __name__ == "__main__":
    unittest.main()
