"""Email bodies, by template name."""

TEMPLATES = {
    "welcome": "Welcome to the shop, {name}!",
    "password_reset": "Reset your password: https://shop.example/reset/{token}",
    "receipt": "Thanks for your order {order_id}.",
}


def render(name, context):
    return TEMPLATES[name].format(**context)
