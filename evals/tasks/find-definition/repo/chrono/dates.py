"""Calendar dates."""
import datetime


def parse_iso_date(text):
    """A date from YYYY-MM-DD."""
    year, month, day = (int(part) for part in text.split("-"))
    return datetime.date(year, month, day)


def is_weekend(date):
    return date.weekday() >= 5
