"""Signing webhook payloads."""
import hashlib
import hmac


def sign(secret, body):
    return hmac.new(secret, body, hashlib.sha256).hexdigest()
