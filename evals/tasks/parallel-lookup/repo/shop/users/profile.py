"""Profile edits."""


def rename(user, name):
    user["name"] = name.strip()
    return user


def change_email(user, email):
    if "@" not in email:
        raise ValueError("not an email address")
    user["email"] = email
    return user
