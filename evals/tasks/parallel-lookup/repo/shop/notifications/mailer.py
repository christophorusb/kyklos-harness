"""The outgoing mail queue."""
from shop.notifications.email_templates import render

OUTBOX = []


def send(to, template, context):
    """Queue an email. Delivery is a separate worker's job."""
    OUTBOX.append({"to": to, "body": render(template, context)})
