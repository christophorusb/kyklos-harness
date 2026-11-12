"""Turning a cart into an order."""
from shop.billing.invoices import create_invoice


def checkout(cart, region):
    order = {"id": len(cart.get("lines", [])), "lines": cart.get("lines", []), "region": region}
    return order, create_invoice(order)
