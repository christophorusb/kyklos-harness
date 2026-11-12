"""Count the lines, words and characters in a file, like a small `wc`."""
import sys


def counts(text):
    return len(text.splitlines()), len(text.split()), len(text)


def main(argv):
    if len(argv) != 1:
        print("usage: wordcount.py FILE", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        lines, words, chars = counts(f.read())
    print("%d %d %d" % (lines, words, chars))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
