"""Sums on an invoice."""

TAX_RATES = {"EU": 0.21, "UK": 0.20, "US": 0.0}


def subtotal(invoice):
    return sum(line["price"] * line["quantity"] for line in invoice["lines"])


def tax_for(invoice):
    """Tax owed on an invoice, at its region's rate, rounded to the cent."""
    return round(subtotal(invoice) * TAX_RATES.get(invoice["region"], 0.0), 2)
