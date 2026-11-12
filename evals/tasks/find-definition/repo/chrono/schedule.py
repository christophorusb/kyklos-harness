"""Recurring jobs."""
from chrono.convert import clamp_interval, to_seconds


def interval_for(job):
    """How often a job runs, in seconds, from its ISO 8601 `every` field."""
    return clamp_interval(to_seconds(job["every"]))
