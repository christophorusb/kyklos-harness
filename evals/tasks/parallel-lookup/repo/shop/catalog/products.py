"""The product list."""

PRODUCTS = {}


def add(sku, name, price):
    PRODUCTS[sku] = {"sku": sku, "name": name, "price": price}


def find(sku):
    return PRODUCTS.get(sku)
