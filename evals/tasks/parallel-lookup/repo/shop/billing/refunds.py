"""Refunds."""


def refund_amount(invoice, lines):
    return sum(line["price"] * line["quantity"] for line in lines)
