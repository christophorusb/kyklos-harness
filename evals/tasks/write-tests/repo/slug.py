"""Turn titles into URL slugs."""
import re
import unicodedata


def slugify(text, sep="-"):
    """Lowercase ASCII words from `text`, joined by `sep`: accents are dropped, and anything
    that isn't a letter or a digit separates words."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words = re.split(r"[^a-z0-9]+", ascii_text.lower())
    return sep.join(word for word in words if word)
