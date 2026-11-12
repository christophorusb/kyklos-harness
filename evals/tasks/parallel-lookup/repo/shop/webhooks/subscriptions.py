"""Who wants which events."""

SUBSCRIPTIONS = []


def subscribe(url, events):
    SUBSCRIPTIONS.append({"url": url, "events": set(events)})


def for_event(name):
    return [s for s in SUBSCRIPTIONS if name in s["events"]]
