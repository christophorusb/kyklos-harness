"""Formats from the old scheduler, kept for imports of old data."""


def to_seconds_legacy(spec):
    """Seconds from the old HH:MM:SS clock format."""
    hours, minutes, seconds = (int(part) for part in spec.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def from_seconds_legacy(total):
    hours, rest = divmod(int(total), 3600)
    minutes, seconds = divmod(rest, 60)
    return "%02d:%02d:%02d" % (hours, minutes, seconds)
