from checklib import fail, ok, python, unchanged

unchanged("test_pager.py")
code, out = python("-m", "unittest", "-q", "test_pager")
if code:
    fail("the tests still fail:\n" + out[-1500:])
ok()
