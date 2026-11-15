"""Image generation for the generate_image tool: one prompt to an image model, one image back.

The image model is a second provider, configured on its own (IMAGE_BASE_URL, IMAGE_API_KEY,
IMAGE_MODEL) and never borrowing the chat provider's settings: the same rule profiles follow, so one
provider's key can't reach another's host. Two request shapes cover the OpenAI-compatible world:

  IMAGE_API=images  POST /images/generations, the OpenAI Images API (OpenAI, xAI, Together, ...).
                    The image comes back as b64_json, or as a URL that is fetched once over https.
  IMAGE_API=chat    POST /chat/completions with modalities ["image", "text"], as OpenRouter serves
                    image models; the image comes back in message.images as a data URL.

This module only produces bytes. Checking them, choosing the file and asking permission are the
caller's (Agent.generate_image), so nothing here touches the disk.
"""

import base64
import re
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

IMAGE_APIS = ("images", "chat")
MAX_GENERATED_BYTES = 20_000_000  # a 4K PNG fits; anything larger is not an image we want on disk
REQUEST_TIMEOUT = 180  # seconds: image models are slow, and a stuck request should still end
MAX_PROMPT_CHARS = 4000
SIZE_RE = re.compile(r"^\d{2,4}x\d{2,4}$")
DATA_URL_RE = re.compile(r"^data:image/[\w.+-]+;base64,(.+)$", re.S)


class ImageError(Exception):
    """Generation failed, with a reason worth showing the model and the person."""


@dataclass(frozen=True)
class ImageConfig:
    base_url: str
    api_key: str
    model: str
    api: str = "images"
    size: str | None = None  # the default size to ask for; None lets the provider choose

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or self.base_url


def image_config(env) -> ImageConfig | None:
    """The image provider from the environment, or None if none is set up.

    Raises ValueError for a half-finished setup, so a typo is reported at startup instead of the
    tool quietly not existing.
    """
    names = ("IMAGE_BASE_URL", "IMAGE_API_KEY", "IMAGE_MODEL")
    values = {name: (env.get(name) or "").strip() for name in names}
    if not any(values.values()):
        return None
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Image generation needs {', '.join(names)} set together; missing {', '.join(missing)}.")
    api = (env.get("IMAGE_API") or "images").strip().lower()
    if api not in IMAGE_APIS:
        raise ValueError(f"IMAGE_API must be one of {', '.join(IMAGE_APIS)}, not {api!r}.")
    size = (env.get("IMAGE_SIZE") or "").strip() or None
    if size and not SIZE_RE.match(size):
        raise ValueError(f"IMAGE_SIZE must look like 1024x1024, not {size!r}.")
    return ImageConfig(values["IMAGE_BASE_URL"], values["IMAGE_API_KEY"], values["IMAGE_MODEL"], api, size)


def _decode(b64: str) -> bytes:
    try:
        data = base64.b64decode(b64, validate=False)
    except (ValueError, TypeError) as e:
        raise ImageError(f"The image provider sent data that isn't valid base64 ({e}).")
    if len(data) > MAX_GENERATED_BYTES:
        raise ImageError(f"The generated image is {len(data):,} bytes, over the {MAX_GENERATED_BYTES:,}-byte limit.")
    return data


def _download(url: str, opener=urllib.request.urlopen) -> bytes:
    """Fetch an image the provider returned by URL: https only, with a size cap and a timeout."""
    if urlparse(url).scheme != "https":
        raise ImageError("The image provider returned a non-https URL, which Kyklos won't fetch.")
    try:
        with opener(url, timeout=REQUEST_TIMEOUT) as response:
            data = response.read(MAX_GENERATED_BYTES + 1)
    except OSError as e:
        raise ImageError(f"Couldn't download the generated image: {e}")
    if len(data) > MAX_GENERATED_BYTES:
        raise ImageError(f"The generated image is over the {MAX_GENERATED_BYTES:,}-byte limit.")
    return data


def _first_image_url(message) -> str | None:
    """The first image in a chat reply: message.images (OpenRouter), or an image_url content part."""
    images = getattr(message, "images", None)
    if images is None and hasattr(message, "model_extra"):
        images = (message.model_extra or {}).get("images")
    for part in list(images or []) + (list(message.content) if isinstance(getattr(message, "content", None), list) else []):
        part = part if isinstance(part, dict) else getattr(part, "model_dump", lambda: {})()
        url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None
        if url:
            return url
    return None


class ImageGenerator:
    def __init__(self, config: ImageConfig, client=None, opener=None):
        self.config = config
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=REQUEST_TIMEOUT, max_retries=1)
        self.client = client
        self.opener = opener or urllib.request.urlopen

    def generate(self, prompt: str, size: str | None = None) -> bytes:
        """The image's bytes, as the provider sent them. Raises ImageError."""
        size = size or self.config.size
        try:
            if self.config.api == "images":
                return self._via_images(prompt, size)
            return self._via_chat(prompt, size)
        except ImageError:
            raise
        except Exception as e:  # the SDK's API errors, timeouts, connection failures
            raise ImageError(f"The image provider ({self.config.host}) refused or failed: {e}")

    def _via_images(self, prompt: str, size: str | None) -> bytes:
        kwargs = {"model": self.config.model, "prompt": prompt, "n": 1}
        if size:
            kwargs["size"] = size
        response = self.client.images.generate(**kwargs)
        data = getattr(response, "data", None) or []
        if not data:
            raise ImageError("The image provider answered without an image.")
        first = data[0]
        if getattr(first, "b64_json", None):
            return _decode(first.b64_json)
        if getattr(first, "url", None):
            return _download(first.url, self.opener)
        raise ImageError("The image provider answered without an image.")

    def _via_chat(self, prompt: str, size: str | None) -> bytes:
        text = prompt if not size else f"{prompt}\n\n(Image size: {size}.)"
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": text}],
            extra_body={"modalities": ["image", "text"]},
        )
        choices = getattr(response, "choices", None) or []
        url = _first_image_url(choices[0].message) if choices else None
        if not url:
            said = (getattr(choices[0].message, "content", None) or "").strip() if choices else ""
            raise ImageError("The image model replied without an image"
                             + (f": {said[:300]}" if isinstance(said, str) and said else "."))
        match = DATA_URL_RE.match(url)
        if match:
            return _decode(match.group(1))
        return _download(url, self.opener)
