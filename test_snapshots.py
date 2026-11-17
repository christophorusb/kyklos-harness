"""Offline tests for workspace snapshots: /undo reverting what commands changed, not only file tools.

A scripted fake model drives the real loop; its bash calls run for real in a temporary workspace.

Run:  python -m unittest test_snapshots -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_kyklos import FakeClient, allow_all, reply, tool_call  # noqa: F401 (sets KYKLOS_SNAPSHOTS=0)

import kyklos
from kyklos import Agent, CommandContext, run_command
from snapshots import WorkspaceSnapshots

HAS_GIT = shutil.which("git") is not None
POSIX = sys.platform != "win32"


def bash_turn(*commands, answer="Done."):
    """A turn that runs each command in turn, then answers."""
    script = [reply(tool_calls=[tool_call(str(i), "bash", command=c)]) for i, c in enumerate(commands)]
    return script + [reply(content=answer)]


@unittest.skipUnless(HAS_GIT and POSIX, "needs git and a POSIX shell")
class UndoRevertsCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name).resolve()
        (self.ws / "app.py").write_text("x = 1\n")
        (self.ws / "keep.txt").write_text("keep\n")

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, script, **kw):
        agent = Agent(FakeClient(script), "fake", self.ws, permission=allow_all,
                      workspace_snapshots=True, **kw)
        self.addCleanup(lambda: agent.snapshots and agent.snapshots.close())
        return agent

    def test_edits_creations_and_deletions_by_bash_come_back(self):
        agent = self.agent(bash_turn("sed -i.bak 's/1/2/' app.py && rm app.py.bak",
                                     "mkdir -p gen && echo made > gen/out.txt", "rm keep.txt"))
        agent.send("change things with commands")
        self.assertEqual((self.ws / "app.py").read_text(), "x = 2\n")
        self.assertFalse((self.ws / "keep.txt").exists())

        self.assertEqual(agent.undo(), ["app.py", "gen/out.txt", "keep.txt"])
        self.assertEqual((self.ws / "app.py").read_text(), "x = 1\n")
        self.assertEqual((self.ws / "keep.txt").read_text(), "keep\n")
        self.assertFalse((self.ws / "gen" / "out.txt").exists())
        self.assertEqual(agent.last_undo, {"left_alone": [], "uncovered": None})

    def test_file_tools_and_bash_in_one_turn(self):
        agent = self.agent([
            reply(tool_calls=[tool_call("1", "write_file", path="new.py", content="a = 1\n")]),
            reply(tool_calls=[tool_call("2", "bash", command="echo 'b = 2' >> new.py")]),
            reply(content="Done."),
        ])
        agent.send("write then append")
        self.assertEqual(agent.undo(), ["new.py"])
        self.assertFalse((self.ws / "new.py").exists())

    def test_a_file_you_changed_after_the_turn_is_left_alone(self):
        agent = self.agent(bash_turn("echo agent > app.py", "echo agent > keep.txt"))
        agent.send("overwrite two files")
        (self.ws / "keep.txt").write_text("mine now\n")  # you edit one before undoing

        self.assertEqual(agent.undo(), ["app.py"])
        self.assertEqual((self.ws / "app.py").read_text(), "x = 1\n")
        self.assertEqual((self.ws / "keep.txt").read_text(), "mine now\n")
        self.assertEqual(agent.last_undo["left_alone"], ["keep.txt"])

    def test_undo_command_names_what_it_left_alone(self):
        agent = self.agent(bash_turn("echo agent > keep.txt"))
        agent.send("overwrite")
        (self.ws / "keep.txt").write_text("mine\n")
        said = []
        run_command(CommandContext(agent, lambda kind, text: said.append((kind, text)),
                                   self.ws / "s.jsonl"), "/undo")
        self.assertIn(("warn", "Left alone, since they changed again after that turn: keep.txt"), said)

    def test_turns_undo_newest_first(self):
        agent = self.agent(bash_turn("echo 2 > app.py") + bash_turn("echo 3 > app.py"))
        agent.send("first")
        agent.send("second")
        agent.undo()
        self.assertEqual((self.ws / "app.py").read_text(), "2\n")
        agent.undo()
        self.assertEqual((self.ws / "app.py").read_text(), "x = 1\n")

    def test_changes_between_turns_are_not_the_turns_to_undo(self):
        agent = self.agent(bash_turn("echo agent > app.py"))
        (self.ws / "yours.txt").write_text("before the turn\n")
        agent.send("go")
        self.assertEqual(agent.undo(), ["app.py"])
        self.assertTrue((self.ws / "yours.txt").exists())

    def test_bytes_and_permissions_survive_a_round_trip(self):
        (self.ws / ".gitattributes").write_text("* text=auto eol=lf\n")
        (self.ws / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
        (self.ws / "run.sh").write_text("#!/bin/sh\necho hi\n")
        (self.ws / "run.sh").chmod(0o750)
        (self.ws / "private.txt").write_text("p\n")
        (self.ws / "private.txt").chmod(0o600)
        agent = self.agent(bash_turn("rm crlf.txt run.sh", "echo changed > private.txt"))
        agent.send("delete and change")
        agent.undo()
        self.assertEqual((self.ws / "crlf.txt").read_bytes(), b"one\r\ntwo\r\n")
        self.assertTrue(os.access(self.ws / "run.sh", os.X_OK))
        self.assertEqual((self.ws / "private.txt").stat().st_mode & 0o777, 0o600)

    def test_a_symlink_comes_back_as_a_symlink(self):
        os.symlink("app.py", self.ws / "link")
        agent = self.agent(bash_turn("rm link && echo plain > link"))
        agent.send("replace the link")
        agent.undo()
        self.assertTrue((self.ws / "link").is_symlink())
        self.assertEqual(os.readlink(self.ws / "link"), "app.py")

    def test_secret_files_are_never_copied_or_touched(self):
        (self.ws / ".env").write_text("API_KEY=sk-do-not-copy\n")
        (self.ws / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
        agent = self.agent(bash_turn("echo x >> app.py"))
        agent.send("go")
        blobs = subprocess.run(
            [agent.snapshots.git, "--git-dir", str(agent.snapshots.store),
             "cat-file", "--batch-all-objects", "--batch"],
            capture_output=True, env=agent.snapshots._env()).stdout
        self.assertNotIn(b"sk-do-not-copy", blobs)
        self.assertNotIn(b"PRIVATE KEY", blobs)
        self.assertIn(b"x = 1", blobs)

    def test_skipped_folders_stay_out(self):
        (self.ws / "node_modules").mkdir()
        (self.ws / "node_modules" / "dep.js").write_text("dep\n")
        agent = self.agent(bash_turn("rm -r node_modules"))
        agent.send("clean")
        self.assertEqual(agent.undo(), [])
        self.assertFalse((self.ws / "node_modules").exists())

    def test_the_file_tools_record_still_covers_what_snapshots_cant_see(self):
        (self.ws / ".gitignore").write_text("ignored.log\n")
        agent = self.agent([
            reply(tool_calls=[tool_call("1", "write_file", path="ignored.log", content="log\n")]),
            reply(content="Done."),
        ])
        agent.send("write an ignored file")
        self.assertEqual(agent.undo(), ["ignored.log"])
        self.assertFalse((self.ws / "ignored.log").exists())

    def test_the_workspaces_own_git_is_untouched(self):
        subprocess.run(["git", "init", "-q", str(self.ws)], check=True)
        before = sorted(p.name for p in (self.ws / ".git").iterdir())
        agent = self.agent(bash_turn("echo y > app.py"))
        agent.send("go")
        agent.undo()
        self.assertEqual(sorted(p.name for p in (self.ws / ".git").iterdir()), before)
        listed = agent.snapshots._git("ls-tree", "-r", "--name-only", agent.snapshots.snapshot())
        self.assertNotIn(b".git/", listed)

    def test_a_subagents_commands_are_part_of_the_turn(self):
        agent = self.agent([
            reply(tool_calls=[tool_call("t", "task", description="d", prompt="p", mode="general")]),
            reply(tool_calls=[tool_call("s", "bash", command="echo sub > sub.txt")]),
            reply(content="Sub done."),
            reply(content="Parent done."),
        ])
        agent.send("delegate")
        self.assertEqual(agent.undo(), ["sub.txt"])


class UndoWithoutSnapshots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name).resolve()
        (self.ws / "app.py").write_text("x = 1\n")

    def tearDown(self):
        self.tmp.cleanup()

    def script(self):
        return [reply(tool_calls=[tool_call("1", "edit_file", path="app.py", old_string="x = 1",
                                            new_string="x = 2")]),
                reply(content="Done.")]

    def test_off_falls_back_to_the_file_tools_and_says_so(self):
        agent = Agent(FakeClient(self.script()), "fake", self.ws, permission=allow_all,
                      workspace_snapshots=False)
        agent.send("edit")
        self.assertEqual(agent.undo(), ["app.py"])
        self.assertEqual((self.ws / "app.py").read_text(), "x = 1\n")
        self.assertIsNone(agent.last_undo["uncovered"])  # no command ran, so nothing was missed
        self.assertIsNone(agent.snapshots)

    @unittest.skipUnless(POSIX, "needs a POSIX shell")
    def test_off_says_when_a_command_ran(self):
        agent = Agent(FakeClient(bash_turn("echo y > app.py")), "fake", self.ws,
                      permission=allow_all, workspace_snapshots=False)
        agent.send("run")
        self.assertEqual(agent.undo(), [])
        self.assertEqual((self.ws / "app.py").read_text(), "y\n")
        self.assertEqual(agent.last_undo["uncovered"], "KYKLOS_SNAPSHOTS is off")

    def test_without_git_the_file_tools_still_undo(self):
        with mock.patch("snapshots.shutil.which", return_value=None):
            agent = Agent(FakeClient(self.script()), "fake", self.ws, permission=allow_all,
                          workspace_snapshots=True)
            agent.send("edit")
        self.assertEqual(agent.undo(), ["app.py"])
        self.assertEqual(agent.snapshots_off, "git isn't installed")

    def test_home_folder_is_not_snapshotted(self):
        with mock.patch.object(kyklos, "is_broad_folder", return_value=True):
            agent = Agent(FakeClient(self.script()), "fake", self.ws, permission=allow_all,
                          workspace_snapshots=True)
            agent.send("edit")
        self.assertIsNone(agent.snapshots)
        self.assertIn("home folder", agent.snapshots_off)

    @unittest.skipUnless(HAS_GIT, "needs git")
    def test_a_failing_snapshot_turns_itself_off_with_a_notice(self):
        events = []
        agent = Agent(FakeClient(self.script()), "fake", self.ws, permission=allow_all,
                      workspace_snapshots=True, on_event=lambda e, **d: events.append((e, d)))
        self.addCleanup(lambda: agent.snapshots and agent.snapshots.close())
        with mock.patch.object(WorkspaceSnapshots, "_git",
                               side_effect=kyklos.SnapshotError("git took over 30s")):
            agent.send("edit")
        notices = [d["text"] for e, d in events if e == "notice"]
        self.assertTrue(any("git took over 30s" in n for n in notices))
        self.assertEqual(agent.undo(), ["app.py"])  # the file tools' record still works

    def test_setting_parses(self):
        self.assertTrue(kyklos.choose_snapshots({}))
        self.assertFalse(kyklos.choose_snapshots({"KYKLOS_SNAPSHOTS": "0"}))
        with self.assertRaises(ValueError):
            kyklos.choose_snapshots({"KYKLOS_SNAPSHOTS": "maybe"})


@unittest.skipUnless(HAS_GIT and POSIX, "needs git and symlinks")
class RestoreStaysInside(unittest.TestCase):
    def test_a_path_behind_a_link_out_of_the_workspace_is_left_alone(self):
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            ws, outside = Path(ws).resolve(), Path(outside).resolve()
            (ws / "d").mkdir()
            (ws / "d" / "f.txt").write_text("orig\n")
            snaps = WorkspaceSnapshots.open(ws)
            self.addCleanup(snaps.close)
            start = snaps.snapshot()
            (ws / "d" / "f.txt").write_text("changed\n")
            end = snaps.snapshot()
            shutil.rmtree(ws / "d")
            os.symlink(outside, ws / "d")  # something swapped the folder for a way out
            self.assertIsNone(snaps._target("d/f.txt"))
            self.assertIsNotNone(snaps._target("e/f.txt"))  # a folder that doesn't exist yet is fine
            reverted, left_alone = snaps.restore(start, end)
            self.assertEqual(reverted, [])
            self.assertEqual(left_alone, ["d/f.txt"])
            self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
