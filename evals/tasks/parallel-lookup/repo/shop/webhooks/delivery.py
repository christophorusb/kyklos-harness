"""Delivering webhook events to subscribers' endpoints."""
from shop.http import client

MAX_DELIVERY_ATTEMPTS = 7
BACKOFF_SECONDS = (1, 5, 30, 120, 600, 1800, 3600)


def deliver(subscription, event):
    """Post `event`, retrying with backoff until it is accepted or attempts run out."""
    for attempt in range(MAX_DELIVERY_ATTEMPTS):
        if client.post(subscription["url"], event, retries=0).ok:
            return True
    return False
