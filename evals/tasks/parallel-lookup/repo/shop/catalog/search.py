"""Product search."""
from shop.catalog.products import PRODUCTS


def search(text):
    text = text.lower()
    return [p for p in PRODUCTS.values() if text in p["name"].lower()]
