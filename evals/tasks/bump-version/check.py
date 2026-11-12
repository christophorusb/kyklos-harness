import re

from checklib import WS, fail, ok


def read(rel):
    return (WS / rel).read_text(encoding="utf-8")


if not re.search(r'^version = "1\.5\.0"$', read("pyproject.toml"), re.M):
    fail("pyproject.toml doesn't say version = \"1.5.0\"")
if '__version__ = "1.5.0"' not in read("src/tinypkg/__init__.py"):
    fail("__version__ is not 1.5.0")
if "tinypkg==1.5.0" not in read("README.md") or "1.4.2" in read("README.md"):
    fail("README.md doesn't pin 1.5.0")
log = read("CHANGELOG.md")
heads = re.findall(r"^## (.+)$", log, re.M)
if heads != ["1.5.0", "1.4.2", "1.4.1"]:
    fail("changelog headings are %r, expected 1.5.0, 1.4.2, 1.4.1" % heads)
for kept in ("- Faster parsing of long inputs.", "- Fixed a crash on empty input (reported against 1.4.1).",
             "- Documentation fixes."):
    if kept not in log:
        fail("the changelog lost or changed the entry %r" % kept)
first = log.split("## 1.4.2")[0]
if "Faster parsing" not in first:
    fail("the unreleased entries didn't stay under 1.5.0")
ok()
