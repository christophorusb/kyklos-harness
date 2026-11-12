"""In-memory user store."""

_USERS = {
    1: {"name": "Ada", "email": "ada@example.com"},
    2: {"name": "Lin", "email": "lin@example.com"},
}


def get_usr(uid):
    """The user with id `uid`, or None."""
    return _USERS.get(uid)


def get_usr_email(uid):
    """The email address of user `uid`, or None."""
    user = get_usr(uid)
    return user["email"] if user else None
