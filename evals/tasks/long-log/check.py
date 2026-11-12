from checklib import fail, ok, python, unchanged

unchanged("run_checks.py")
code, out = python("run_checks.py")
if code:
    fail("run_checks.py still fails: " + out.strip().splitlines()[-1])
code, out = python("-c", "from rangeutil import clamp\n"
                         "assert clamp(9, 0, 3) == 3 and clamp(-1, 0, 3) == 0 and clamp(2, 0, 3) == 2")
if code:
    fail("clamp is not right for values above, below and inside the range:\n" + out[-800:])
ok()
