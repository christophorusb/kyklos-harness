"""Offline tests for image generation: the generate_image tool, its client, and its permission prompt.

Fake image providers stand in for the real ones; the agent loop around them is the real one.

Run:  python -m unittest test_imagegen -v
"""

import base64
import json
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import kyklos
from imagegen import ImageConfig, ImageError, ImageGenerator, image_config
from kyklos import IMAGE_SCHEMA, SUBAGENT_TOOLS, Agent, permission_options
from test_kyklos import FakeClient, allow_all, reply, tool_call


def png(width=2, height=2) -> bytes:
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
CONFIG = ImageConfig("https://images.example/v1", "sk-image-test", "pixel-model")


class FakeImagesAPI:
    """client.images.generate and client.chat.completions.create, answering with scripted results."""

    def __init__(self, result=None, chat_message=None, error=None):
        self.result, self.chat_message, self.error = result, chat_message, error
        self.calls = []
        self.images = SimpleNamespace(generate=self._generate)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(data=[self.result] if self.result else [])

    def _chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=self.chat_message)])


def b64_result(data: bytes):
    return SimpleNamespace(b64_json=base64.b64encode(data).decode(), url=None)


class ConfigTests(unittest.TestCase):
    def test_nothing_set_means_no_image_tool(self):
        self.assertIsNone(image_config({}))

    def test_a_half_finished_setup_is_an_error_not_a_silent_no(self):
        with self.assertRaises(ValueError) as caught:
            image_config({"IMAGE_MODEL": "x", "IMAGE_BASE_URL": "https://a.example/v1"})
        self.assertIn("IMAGE_API_KEY", str(caught.exception))

    def test_bad_api_and_size_are_refused(self):
        full = {"IMAGE_BASE_URL": "https://a.example/v1", "IMAGE_API_KEY": "k", "IMAGE_MODEL": "m"}
        with self.assertRaises(ValueError):
            image_config({**full, "IMAGE_API": "magic"})
        with self.assertRaises(ValueError):
            image_config({**full, "IMAGE_SIZE": "big"})

    def test_a_full_setup(self):
        cfg = image_config({"IMAGE_BASE_URL": "https://a.example/v1", "IMAGE_API_KEY": "k", "IMAGE_MODEL": "m",
                            "IMAGE_API": "chat", "IMAGE_SIZE": "512x512"})
        self.assertEqual((cfg.api, cfg.size, cfg.host), ("chat", "512x512", "a.example"))

    def test_the_chat_profile_is_never_borrowed(self):
        # BASE_URL / API_KEY belong to the chat provider; the image block must stand on its own.
        self.assertIsNone(image_config({"BASE_URL": "https://chat.example", "API_KEY": "k", "MODEL": "m"}))


class GeneratorTests(unittest.TestCase):
    def test_images_api_base64(self):
        api = FakeImagesAPI(result=b64_result(png()))
        data = ImageGenerator(CONFIG, client=api).generate("a red square", "256x256")
        self.assertEqual(data, png())
        self.assertEqual(api.calls[0], {"model": "pixel-model", "prompt": "a red square", "n": 1, "size": "256x256"})

    def test_images_api_leaves_size_to_the_provider_when_unset(self):
        api = FakeImagesAPI(result=b64_result(png()))
        ImageGenerator(CONFIG, client=api).generate("x")
        self.assertNotIn("size", api.calls[0])

    def test_a_url_is_fetched_over_https_only(self):
        opened = []

        def opener(url, timeout):
            opened.append(url)
            return mock.MagicMock(__enter__=lambda s: SimpleNamespace(read=lambda n: png()), __exit__=lambda *a: None)

        api = FakeImagesAPI(result=SimpleNamespace(b64_json=None, url="https://cdn.example/i.png"))
        self.assertEqual(ImageGenerator(CONFIG, client=api, opener=opener).generate("x"), png())
        self.assertEqual(opened, ["https://cdn.example/i.png"])
        api = FakeImagesAPI(result=SimpleNamespace(b64_json=None, url="http://cdn.example/i.png"))
        with self.assertRaises(ImageError):
            ImageGenerator(CONFIG, client=api, opener=opener).generate("x")

    def test_oversized_images_are_refused(self):
        api = FakeImagesAPI(result=b64_result(b"\0" * 21_000_000))
        with self.assertRaises(ImageError):
            ImageGenerator(CONFIG, client=api).generate("x")

    def test_chat_api_reads_message_images(self):
        url = "data:image/png;base64," + base64.b64encode(png()).decode()
        message = SimpleNamespace(content="Here it is.", images=[{"type": "image_url", "image_url": {"url": url}}])
        api = FakeImagesAPI(chat_message=message)
        chat_cfg = ImageConfig(CONFIG.base_url, CONFIG.api_key, CONFIG.model, api="chat")
        self.assertEqual(ImageGenerator(chat_cfg, client=api).generate("x"), png())
        self.assertEqual(api.calls[0]["extra_body"], {"modalities": ["image", "text"]})

    def test_chat_api_without_an_image_says_what_the_model_said(self):
        api = FakeImagesAPI(chat_message=SimpleNamespace(content="I can't draw that.", images=None))
        chat_cfg = ImageConfig(CONFIG.base_url, CONFIG.api_key, CONFIG.model, api="chat")
        with self.assertRaises(ImageError) as caught:
            ImageGenerator(chat_cfg, client=api).generate("x")
        self.assertIn("I can't draw that.", str(caught.exception))

    def test_provider_errors_name_the_host(self):
        api = FakeImagesAPI(error=RuntimeError("401 bad key"))
        with self.assertRaises(ImageError) as caught:
            ImageGenerator(CONFIG, client=api).generate("x")
        self.assertIn("images.example", str(caught.exception))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.events = []
        self.asked = []

    def tearDown(self):
        self.tmp.cleanup()
        kyklos.set_image_provider(None, None)

    def agent(self, replies, image_bytes=None, permission=None, api=None):
        self.api = api or FakeImagesAPI(result=b64_result(image_bytes if image_bytes is not None else png()))

        def ask(name, args, always):
            self.asked.append((name, args))
            return (permission or allow_all)(name, args, always)

        return Agent(FakeClient(replies), "chat-model", self.ws, permission=ask,
                     on_event=lambda event, depth=0, **data: self.events.append((event, data)),
                     images=ImageGenerator(CONFIG, client=self.api))

    def call(self, **args):
        return [reply(tool_calls=[tool_call("g1", "generate_image", **args)]), reply(content="Done.")]

    def result(self, agent):
        return [m for m in agent.messages if m["role"] == "tool"][-1]["content"]

    def test_offered_only_when_configured_and_never_in_plan_mode_or_subagents(self):
        plain = Agent(FakeClient([]), "m", self.ws)
        self.assertNotIn("generate_image", plain.tool_names)
        agent = self.agent([])
        self.assertIn("generate_image", agent.tool_names)
        agent.set_plan_mode(True)
        self.assertNotIn("generate_image", agent.tool_names)
        for mode, names in SUBAGENT_TOOLS.items():
            self.assertNotIn("generate_image", names, mode)

    def test_generates_saves_and_reports(self):
        agent = self.agent(self.call(prompt="a knight sprite", path="assets/hero.png", size="64x64"))
        self.assertEqual(agent.send("make a hero sprite"), "Done.")
        self.assertEqual((self.ws / "assets/hero.png").read_bytes(), png())
        self.assertIn("Generated assets/hero.png (PNG, 2×2", self.result(agent))
        self.assertEqual(self.api.calls[0]["prompt"], "a knight sprite")
        self.assertEqual(self.api.calls[0]["size"], "64x64")
        image = [data for event, data in self.events if event == "image"]
        self.assertEqual((image[0]["rel"], image[0]["mime"], image[0]["width"]), ("assets/hero.png", "image/png", 2))

    def test_every_call_asks_and_a_denial_sends_nothing(self):
        agent = self.agent(self.call(prompt="x", path="a.png"), permission=lambda *a: (False, "denied by test"))
        agent.send("go")
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self.api.calls, [], "a denied call must not reach the image provider")
        self.assertFalse((self.ws / "a.png").exists())

    def test_bad_arguments_are_refused_before_anyone_is_asked(self):
        (self.ws / ".git").mkdir()
        for args in ({"prompt": "x", "path": "../outside.png"}, {"prompt": "x", "path": ".git/x.png"},
                     {"prompt": "x", "path": "notes.txt"}, {"prompt": "", "path": "a.png"},
                     {"prompt": "x", "path": "a.png", "size": "huge"}, {"prompt": "x" * 5000, "path": "a.png"}):
            with self.subTest(args=args):
                self.asked.clear()
                agent = self.agent(self.call(**args))
                agent.send("go")
                self.assertEqual(self.asked, [])
                self.assertEqual(self.api.calls, [])
                self.assertTrue(self.result(agent).startswith("Error"), self.result(agent))

    def test_the_extension_follows_the_content(self):
        agent = self.agent(self.call(prompt="x", path="art/cover.png"), image_bytes=JPEG)
        agent.send("go")
        self.assertFalse((self.ws / "art/cover.png").exists())
        self.assertEqual((self.ws / "art/cover.jpg").read_bytes(), JPEG)
        self.assertIn("saved as art/cover.jpg", self.result(agent))

    def test_non_image_bytes_are_not_saved(self):
        agent = self.agent(self.call(prompt="x", path="a.png"), image_bytes=b"<svg onload=alert(1)>")
        agent.send("go")
        self.assertFalse((self.ws / "a.png").exists())
        self.assertIn("nothing was saved", self.result(agent))

    def test_secret_values_never_reach_the_image_provider(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_supersecretvalue1234567890abcd"}):
            agent = self.agent(self.call(prompt="logo with text ghp_supersecretvalue1234567890abcd", path="a.png"))
            agent.send("go")
        self.assertNotIn("ghp_supersecretvalue1234567890abcd", self.api.calls[0]["prompt"])
        self.assertIn("removed 1 secret value", self.result(agent))

    def test_undo_removes_the_generated_image(self):
        agent = self.agent(self.call(prompt="x", path="a.png"))
        agent.send("go")
        self.assertTrue((self.ws / "a.png").exists())
        agent.undo()
        self.assertFalse((self.ws / "a.png").exists())

    def test_the_permission_prompt_says_where_the_prompt_goes(self):
        self.agent([])  # sets the provider the prompt names
        kyklos.set_explain_workspace(self.ws)
        (self.ws / "existing.png").write_bytes(png())
        touches_secret, prefix, findings = permission_options("generate_image", {"prompt": "x", "path": "existing.png"})
        self.assertEqual((touches_secret, prefix), (False, ""))
        text = " ".join(t for _, t in findings)
        self.assertIn("images.example", text)
        self.assertIn("pixel-model", text)
        self.assertIn("overwrites existing.png", text)
        _, _, findings = permission_options("generate_image", {"prompt": "x", "path": ".github/workflows/x.png"})
        self.assertTrue(any(level == "danger" for level, _ in findings))

    def test_always_allow_is_offered_but_only_by_the_person(self):
        agent = self.agent(self.call(prompt="x", path="a.png") + self.call(prompt="y", path="b.png"),
                           permission=lambda name, args, always: (always.add(name) or True, ""))
        agent.send("first")
        agent.send("second")
        self.assertEqual(len(self.asked), 1, "after 'always', the second call runs without asking")
        self.assertTrue((self.ws / "b.png").exists())

    def test_the_schema_requires_a_prompt_and_a_path(self):
        self.assertEqual(IMAGE_SCHEMA["function"]["parameters"]["required"], ["prompt", "path"])
        json.dumps(IMAGE_SCHEMA)  # serializable as sent to the provider


if __name__ == "__main__":
    unittest.main()
