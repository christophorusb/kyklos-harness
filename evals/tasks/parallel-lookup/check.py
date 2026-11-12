import re

from checklib import answer, fail, ok, unchanged

unchanged()
said = answer()
wanted = {"users/recovery.py": "the module that sends the reset email",
          "webhooks/delivery.py": "where the webhook attempt limit is defined",
          "billing/totals.py": "the file with the invoice tax function",
          "tax_for": "the invoice tax function's name"}
for needle, what in wanted.items():
    if needle not in said:
        fail("the answer doesn't give %s (%s)" % (what, needle))
if not re.search(r"(?<!\d)7(?!\d)", said):
    fail("the answer doesn't give the attempt limit, 7")
ok()
