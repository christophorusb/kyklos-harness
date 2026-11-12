"""Showing durations to people."""


def format_duration(seconds):
    """1h30m style: hours and minutes, dropping zero parts."""
    hours, rest = divmod(int(seconds), 3600)
    minutes = rest // 60
    parts = []
    if hours:
        parts.append("%dh" % hours)
    if minutes or not parts:
        parts.append("%dm" % minutes)
    return "".join(parts)


def humanize(seconds):
    """Rough words for a duration: 'a few seconds', 'about 3 minutes', ..."""
    if seconds < 45:
        return "a few seconds"
    if seconds < 3600:
        return "about %d minutes" % round(seconds / 60)
    return "about %d hours" % round(seconds / 3600)
