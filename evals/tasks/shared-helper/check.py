from checklib import fail, ok, python, unchanged

unchanged("test_money.py")
code, out = python("-m", "unittest", "-q", "test_money")
if code:
    fail("the tests fail:\n" + out[-1500:])
ok()
