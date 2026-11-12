import re

from checklib import WS, fail, ok, python, unchanged

unchanged("slug.py")
if not (WS / "test_slug.py").is_file():
    fail("there is no test_slug.py")
code, out = python("-m", "unittest", "-v", "test_slug")
if code:
    fail("the tests fail against the real slugify:\n" + out[-1500:])
if len(re.findall(r" \.\.\. ok$", out, re.M)) < 4:
    fail("fewer than four tests ran:\n" + out[-800:])

# Each mutant breaks one promise the task named; good tests catch every one.
real = (WS / "slug.py").read_text(encoding="utf-8")
mutants = {
    "accents not folded": real.replace(
        'unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")', "text"),
    "empty words kept": real.replace("sep.join(word for word in words if word)", "sep.join(words)"),
    "sep ignored": real.replace("sep.join(word for word in words if word)",
                                '"-".join(word for word in words if word)'),
    "case kept": real.replace('re.split(r"[^a-z0-9]+", ascii_text.lower())',
                              're.split(r"[^A-Za-z0-9]+", ascii_text)'),
}
missed = []
try:
    for name, source in mutants.items():
        if source == real:
            fail("the check's mutant %r no longer applies to slug.py" % name)
        (WS / "slug.py").write_text(source, encoding="utf-8")
        code, _ = python("-m", "unittest", "-q", "test_slug")
        if code == 0:
            missed.append(name)
finally:
    (WS / "slug.py").write_text(real, encoding="utf-8")
if missed:
    fail("the tests pass even when slugify is broken: " + ", ".join(missed))
ok()
