"""Password hashing."""
import hashlib
import os


def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    return salt, hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def check_password(password, salt, digest):
    return hash_password(password, salt)[1] == digest
