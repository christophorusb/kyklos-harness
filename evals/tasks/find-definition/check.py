import re

from checklib import answer, fail, ok, unchanged

LINE = 21
unchanged()
said = answer()
if "convert.py" not in said:
    fail("the answer doesn't name chrono/convert.py")
if not re.search(r"(?<!\d)%d(?!\d)" % LINE, said):
    fail("the answer doesn't give line %d" % LINE)
ok()
