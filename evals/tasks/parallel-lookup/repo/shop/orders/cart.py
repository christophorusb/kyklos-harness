"""Carts."""


def add_line(cart, product, quantity=1):
    cart.setdefault("lines", []).append({"sku": product["sku"], "price": product["price"],
                                         "quantity": quantity})
    return cart
