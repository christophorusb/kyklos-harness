import re

from checklib import WS, fail, ok, python

left = []
for path in sorted(WS.rglob("*")):
    rel = path.relative_to(WS).as_posix()
    if path.is_file() and path.suffix in (".py", ".md") and not rel.startswith(".kyklos/"):
        if re.search(r"\bget_usr\b", path.read_text(encoding="utf-8", errors="replace")):
            left.append(rel)
if left:
    fail("get_usr is still named in " + ", ".join(left))
code, out = python("-c", "from accounts.store import get_user, get_usr_email\n"
                         "assert get_user(1)['name'] == 'Ada'\n"
                         "assert get_usr_email(1) == 'ada@example.com'")
if code:
    fail("get_user and get_usr_email don't both work:\n" + out[-800:])
code, out = python("-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q")
if code:
    fail("the tests fail:\n" + out[-1500:])
ok()
