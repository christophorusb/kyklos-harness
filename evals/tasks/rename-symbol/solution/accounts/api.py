"""Request handlers."""
from accounts.store import get_user, get_usr_email


def show_user(uid):
    user = get_user(uid)
    if user is None:
        return {"status": 404}
    return {"status": 200, "name": user["name"], "email": get_usr_email(uid)}
