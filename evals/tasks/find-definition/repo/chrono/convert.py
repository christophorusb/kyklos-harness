"""Conversions between the units the scheduler stores and the ones its API speaks."""
import re

UNITS = {"D": 86400, "H": 3600, "M": 60, "S": 1}
_PART = re.compile(r"(\d+(?:\.\d+)?)([DHMS])")


def minutes_to_seconds(minutes):
    return int(minutes * 60)


def seconds_to_minutes(seconds):
    return seconds / 60


def clamp_interval(seconds, lowest=60, highest=86400):
    """Keep a polling interval within sane bounds."""
    return max(lowest, min(highest, seconds))


def to_seconds(spec):
    """Seconds in an ISO 8601 duration such as P1D, PT1H30M or PT45S (no years or months)."""
    if not spec.startswith("P"):
        raise ValueError("not an ISO 8601 duration: %r" % spec)
    days, _, clock = spec[1:].partition("T")
    total = 0.0
    for amount, unit in _PART.findall(days):
        total += float(amount) * UNITS[unit]
    for amount, unit in _PART.findall(clock):
        total += float(amount) * UNITS[unit]
    return total
