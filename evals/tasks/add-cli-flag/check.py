import json

from checklib import WS, fail, ok, python

(WS / "check-input.txt").write_text("one two\nthree\n\nfour five six\n", encoding="utf-8")
expected = {"lines": 4, "words": 6, "chars": 29}
code, out = python("wordcount.py", "check-input.txt")
if code or out.strip() != "4 6 29":
    fail("without --json the output changed: %r (exit %d)" % (out.strip(), code))
for argv in (("--json", "check-input.txt"), ("check-input.txt", "--json")):
    code, out = python("wordcount.py", *argv)
    try:
        got = json.loads(out)
    except ValueError:
        fail("%s printed no JSON: %r" % (" ".join(argv), out.strip()[:200]))
    if code or got != expected:
        fail("%s gave %r, expected %r" % (" ".join(argv), got, expected))
code, out = python("wordcount.py")
if code == 0:
    fail("with no file it should still fail with a usage message")
ok()
