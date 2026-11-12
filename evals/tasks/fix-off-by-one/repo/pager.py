"""Split a list into pages."""


def page(items, number, size):
    """Return page `number` (counting from 1) of `items`, with `size` items per page."""
    if number < 1 or size < 1:
        raise ValueError("number and size must be at least 1")
    start = number * size
    return items[start:start + size]


def page_count(items, size):
    """How many pages `items` fills at `size` items per page."""
    if size < 1:
        raise ValueError("size must be at least 1")
    return (len(items) + size - 1) // size
