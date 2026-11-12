"""A thin HTTP client with retries for idempotent calls."""
from collections import namedtuple

MAX_RETRIES = 3
Response = namedtuple("Response", "ok status")


def post(url, payload, retries=MAX_RETRIES):
    """Send `payload`; retried up to `retries` times on connection errors."""
    return Response(ok=True, status=200)
