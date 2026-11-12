"""Shelf prices."""

VAT_INCLUDED_REGIONS = {"EU", "UK"}


def price_with_vat(product, region):
    """The price shown on a product page, VAT included where that region shows it."""
    rate = 0.21 if region == "EU" else 0.20 if region == "UK" else 0.0
    return round(product["price"] * (1 + rate), 2) if region in VAT_INCLUDED_REGIONS else product["price"]
