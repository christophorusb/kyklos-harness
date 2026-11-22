"""Money helpers for the shop. Amounts are stored as integer cents."""


def to_dollars(cents):
    """Cents as a dollar amount."""
    return cents / 100


def format_price(cents):
    """A price for display, such as '$10.50'."""
    return f"${to_dollars(cents):.2f}"


def order_total(items):
    """The total of an order's item prices (in cents), in dollars."""
    return to_dollars(sum(items))


def average_price(items):
    """The mean item price (in cents), in dollars, rounded to the cent."""
    if not items:
        return 0.0
    return round(to_dollars(sum(items)) / len(items), 2)
