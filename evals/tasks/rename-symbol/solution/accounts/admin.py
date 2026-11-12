"""Admin tools."""
from accounts import store


def describe(uid):
    user = store.get_user(uid)
    return "#%d %s" % (uid, user["name"]) if user else "#%d (missing)" % uid
