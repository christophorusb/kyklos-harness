"""Creating invoices."""
from shop.billing import totals


def create_invoice(order):
    invoice = {"order_id": order["id"], "lines": order["lines"], "region": order["region"]}
    invoice["subtotal"] = totals.subtotal(invoice)
    invoice["tax"] = totals.tax_for(invoice)
    invoice["total"] = invoice["subtotal"] + invoice["tax"]
    return invoice
