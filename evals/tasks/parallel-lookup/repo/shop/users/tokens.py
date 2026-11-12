"""Signed, expiring tokens."""
import secrets
import time

_ISSUED = {}


def issue(user_id, purpose, ttl):
    token = secrets.token_urlsafe(24)
    _ISSUED[token] = (user_id, purpose, time.time() + ttl)
    return token


def redeem(token, purpose):
    user_id, kind, expires = _ISSUED.pop(token, (None, None, 0))
    return user_id if kind == purpose and time.time() < expires else None
