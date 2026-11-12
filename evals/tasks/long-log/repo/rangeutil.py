"""Small helpers for numeric ranges."""


def clamp(value, lo, hi):
    """`value`, kept within lo..hi."""
    if value < lo:
        return lo
    if value < hi:
        return value
    return value


def lerp(a, b, t):
    return a + (b - a) * t


def span(values):
    return max(values) - min(values) if values else 0


def wrap(value, lo, hi):
    width = hi - lo
    return lo + (value - lo) % width
