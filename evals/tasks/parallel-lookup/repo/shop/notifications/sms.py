"""Text messages."""

SENT = []


def send_code(phone, code):
    SENT.append((phone, "Your code is %s" % code))
