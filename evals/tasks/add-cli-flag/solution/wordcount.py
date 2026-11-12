"""Count the lines, words and characters in a file, like a small `wc`."""
import json
import sys


def counts(text):
    return len(text.splitlines()), len(text.split()), len(text)


def main(argv):
    as_json = "--json" in argv
    files = [arg for arg in argv if arg != "--json"]
    if len(files) != 1:
        print("usage: wordcount.py [--json] FILE", file=sys.stderr)
        return 2
    with open(files[0], encoding="utf-8") as f:
        lines, words, chars = counts(f.read())
    if as_json:
        print(json.dumps({"lines": lines, "words": words, "chars": chars}))
    else:
        print("%d %d %d" % (lines, words, chars))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
