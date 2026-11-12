"""Account recovery."""
from shop.notifications import mailer
from shop.users import tokens


def send_reset_link(user):
    """Email `user` a link that lets them choose a new password."""
    token = tokens.issue(user["id"], purpose="reset", ttl=3600)
    mailer.send(to=user["email"], template="password_reset", context={"token": token})
