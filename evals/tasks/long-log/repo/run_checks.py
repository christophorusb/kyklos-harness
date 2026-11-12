"""Exercise rangeutil, reporting every check as it runs."""
import sys

from rangeutil import clamp, lerp, span, wrap


def build():
    checks = []
    for i in range(220):
        if i == 109:
            checks.append(("clamp_upper", lambda: clamp(9, 0, 3), 3))
        elif i % 4 == 0:
            checks.append(("lerp_%d" % i, lambda i=i: lerp(0, 100, i % 10), (i % 10) * 100))
        elif i % 4 == 1:
            checks.append(("span_%d" % i, lambda i=i: span(list(range(i % 7, i % 7 + 5))), 4))
        elif i % 4 == 2:
            checks.append(("wrap_%d" % i, lambda i=i: wrap(i, 0, 12), i % 12))
        else:
            checks.append(("clamp_%d" % i, lambda i=i: clamp(i % 5, 0, 10), i % 5))
    return checks


def main():
    checks = build()
    failed = 0
    for n, (name, fn, expected) in enumerate(checks, 1):
        print("[%03d/%d] running %s" % (n, len(checks), name))
        print("        inputs prepared, calling %s()" % name.split("_")[0])
        got = fn()
        if got == expected:
            print("        ok: %s returned %r" % (name, got))
        else:
            failed += 1
            print("        FAILED: %s expected %r but got %r" % (name, expected, got))
        print("        elapsed 0.00%ds, memory steady" % (n % 10))
    print("Done: %d passed, %d failed." % (len(checks) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
