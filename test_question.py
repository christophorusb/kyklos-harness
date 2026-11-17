"""Offline tests for ask_user: the model asking the user a question mid-turn.

A scripted fake model drives the real loop; the answers come from stub answerers, the Aero panel with
typed lines, the plain prompt with a patched input(), and a --serve client over real pipes.

Run:  python -m unittest test_question -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_kyklos import FakeClient, deny_all, reply, tool_call
from test_serve import ServeTestCase
from test_ui import make_ui

import kyklos
from kyklos import (MAX_QUESTIONS_PER_TURN, NO_ANSWER_NOTE, SKIPPED_NOTE, Agent, ask_user_plain,
                    parse_question_answer, tool_overview)
from serve import ServeUI
from ui import describe_call, summarize_result


def ask(call_id="q1", **args):
    return tool_call(call_id, "ask_user", **args)


class Answerer:
    """Records each question and gives the scripted answers in order."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, question, options, multiple):
        self.asked.append((question, options, multiple))
        return self.answers.pop(0)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_turn(self, calls, answerer, **kw):
        """One turn: the model makes `calls` (one message each), then answers. Returns tool results."""
        script = [reply(tool_calls=[c]) for c in calls] + [reply(content="Done.")]
        client = FakeClient(script)
        agent = Agent(client, "fake", self.ws, permission=deny_all, ask=answerer, **kw)
        agent.send("do the thing")
        return [m["content"] for m in agent.messages if m["role"] == "tool"], agent, client

    def test_a_picked_option_comes_back_without_a_permission_prompt(self):
        answerer = Answerer(([1], ""))
        results, _, _ = self.run_turn([ask(question=" Which database? ",
                                           options=["SQLite", " Postgres "])], answerer)
        self.assertEqual(answerer.asked, [("Which database?", ["SQLite", "Postgres"], False)])
        self.assertEqual(results, ['The user chose: "Postgres".'])  # deny_all was never consulted

    def test_own_words_skipped_and_nobody(self):
        for answer, expected in ((([], "use MySQL"), "The user answered in their own words: use MySQL"),
                                 (([], ""), SKIPPED_NOTE), (None, NO_ANSWER_NOTE)):
            results, _, _ = self.run_turn([ask(question="Which?", options=["a", "b"])], Answerer(answer))
            self.assertEqual(results, [expected])

    def test_several_picks_when_multiple(self):
        results, _, _ = self.run_turn([ask(question="Which targets?", options=["mac", "linux", "win"],
                                           multiple=True)], Answerer(([0, 2], "")))
        self.assertEqual(results, ['The user chose: "mac", "win".'])

    def test_bad_arguments_never_reach_the_user(self):
        bad = [dict(question=""), dict(question="Q?", options=["only one"]),
               dict(question="Q?", options=["a", "A"]), dict(question="Q?", multiple=True),
               dict(question="Q?", options=["a", "b"], multiple="yes"),
               dict(question="Q?", options=[f"o{i}" for i in range(7)]),
               dict(question="x" * 1001), dict(question="Q?", default="a")]
        answerer = Answerer()
        results, _, _ = self.run_turn([ask(f"q{i}", **a) for i, a in enumerate(bad)], answerer)
        self.assertEqual(answerer.asked, [])
        self.assertTrue(all(r.startswith("Error:") for r in results), results)

    def test_a_turn_may_ask_only_so_many(self):
        answerer = Answerer(*[([], "ok")] * (MAX_QUESTIONS_PER_TURN + 1))
        calls = [ask(f"q{i}", question=f"Q{i}?") for i in range(MAX_QUESTIONS_PER_TURN + 1)]
        results, _, _ = self.run_turn(calls, answerer)
        self.assertEqual(len(answerer.asked), MAX_QUESTIONS_PER_TURN)
        self.assertIn(f"asked {MAX_QUESTIONS_PER_TURN} questions this turn", results[-1])

    def test_the_count_resets_each_turn(self):
        answerer = Answerer(*[([], "ok")] * (MAX_QUESTIONS_PER_TURN * 2))
        script = []
        for turn in range(2):
            script += [reply(tool_calls=[ask(f"t{turn}q{i}", question="Q?")])
                       for i in range(MAX_QUESTIONS_PER_TURN)] + [reply(content="Done.")]
        agent = Agent(FakeClient(script), "fake", self.ws, ask=answerer)
        agent.send("one")
        agent.send("two")
        self.assertEqual(len(answerer.asked), MAX_QUESTIONS_PER_TURN * 2)

    def test_offered_with_guidance_in_plan_mode_too(self):
        for plan_mode in (False, True):
            _, agent, client = self.run_turn([], Answerer(), plan_mode=plan_mode)
            self.assertIn("ask_user", client.offered[0])
            self.assertIn("ask with ask_user", agent.messages[0]["content"])

    def test_subagents_cant_ask_and_are_told_what_to_do_instead(self):
        script = [reply(tool_calls=[tool_call("t", "task", description="d", prompt="p", mode="general")]),
                  reply(tool_calls=[ask("s", question="Which?")]),
                  reply(content="Sub done."), reply(content="Parent done.")]
        answerer = Answerer()
        client = FakeClient(script)
        agent = Agent(client, "fake", self.ws, ask=answerer)
        agent.send("delegate")
        self.assertNotIn("ask_user", client.offered[1])
        sub_result = [m["content"] for m in client.calls[2] if m["role"] == "tool"][0]
        self.assertIn("Subagents can't reach the user", sub_result)
        self.assertEqual(answerer.asked, [])

    def test_tools_view_lists_it_as_needing_no_approval(self):
        agent = Agent(FakeClient([]), "fake", self.ws)
        self.assertIn("ask_user", tool_overview(agent)["safe"])


class ParseTests(unittest.TestCase):
    def test_numbers_pick_and_anything_else_is_your_words(self):
        opts = ["a", "b", "c"]
        self.assertEqual(parse_question_answer("2", opts, False), ([1], ""))
        self.assertEqual(parse_question_answer(" 3 ", opts, False), ([2], ""))
        self.assertEqual(parse_question_answer("1, 3", opts, True), ([0, 2], ""))
        self.assertEqual(parse_question_answer("1 3 1", opts, True), ([0, 2], ""))
        self.assertEqual(parse_question_answer("1,3", opts, False), ([], "1,3"))  # one pick only
        self.assertEqual(parse_question_answer("4", opts, False), ([], "4"))  # no such option
        self.assertEqual(parse_question_answer("0", opts, False), ([], "0"))
        self.assertEqual(parse_question_answer("2", [], False), ([], "2"))  # an open question
        self.assertEqual(parse_question_answer("b please", opts, False), ([], "b please"))
        self.assertEqual(parse_question_answer("   ", opts, False), ([], ""))


class PlainPromptTests(unittest.TestCase):
    def test_typed_answer_and_no_keyboard(self):
        with mock.patch("builtins.input", return_value="2"), mock.patch("builtins.print"):
            self.assertEqual(ask_user_plain("Which?", ["a", "b"], False), ([1], ""))
        with mock.patch("builtins.input", side_effect=EOFError), mock.patch("builtins.print"):
            self.assertIsNone(ask_user_plain("Which?", ["a", "b"], False))

    def test_eval_runs_have_nobody_to_ask(self):
        self.assertIsNone(kyklos.no_one_to_ask("Q?", [], False))


class AeroTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_panel_shows_the_question_and_records_the_choice(self):
        ui = make_ui(self.ws, lines=["2"])
        self.assertEqual(ui.ask_user("Which database?", ["SQLite", "Postgres"], False), ([1], ""))
        out = ui.console.export_text()
        for text in ("Question", "Which database?", "1  SQLite", "2  Postgres", "chose Postgres"):
            self.assertIn(text, out)

    def test_own_words_and_skip(self):
        ui = make_ui(self.ws, lines=["neither", ""])
        self.assertEqual(ui.ask_user("Which?", ["a", "b"], False), ([], "neither"))
        self.assertEqual(ui.ask_user("Which?", ["a", "b"], False), ([], ""))
        out = ui.console.export_text()
        self.assertIn("answered: neither", out)
        self.assertIn("skipped", out)

    def test_escape_sequences_in_the_question_are_not_rendered(self):
        ui = make_ui(self.ws, lines=["1"])
        ui.ask_user("Pick\x1b]52;c;ZXZpbA==\x07 one", ["a\x1b[2J", "b"], False)
        self.assertNotIn("\x1b]52", ui.console.export_text(styles=True))

    def test_call_line_and_result_summary(self):
        self.assertEqual(describe_call("ask_user", {"question": "Which?\nMore"}), "Which?")
        self.assertEqual(summarize_result("ask_user", {}, 'The user chose: "a".')[1], "answered")
        self.assertEqual(summarize_result("ask_user", {}, SKIPPED_NOTE)[1], "skipped")
        self.assertEqual(summarize_result("ask_user", {}, NO_ANSWER_NOTE)[1], "no one to answer")

    def test_a_turn_through_the_ui(self):
        ui = make_ui(self.ws, lines=["1"])
        client = FakeClient([reply(tool_calls=[ask(question="Tabs or spaces?", options=["tabs", "spaces"])]),
                             reply(content="Tabs it is.")])
        agent = Agent(client, "fake", self.ws, permission=ui.permission, ask=ui.ask_user,
                      on_event=ui.observer)
        ui.agent = agent
        ui.turn(agent, "format it", self.ws / "s.jsonl")
        self.assertIn('The user chose: "tabs".', [m["content"] for m in agent.messages if m["role"] == "tool"])
        self.assertIn("Tabs it is.", ui.console.export_text())


class ServeTests(ServeTestCase):
    def ask_over_the_pipe(self, *replies, multiple=False):
        """Run a turn whose one question is answered with each of `replies` in order."""
        self.requests = []

        def script(c):
            rid = c.call("session/prompt", text="go")
            for answer in replies:
                asked = c.asked("question/request")
                self.requests.append(asked["params"])
                c.answer(asked, **answer)
            c.result(rid)
            self.shutdown(c)

        args = dict(question="Which?", options=["a", "b", "c"]) | ({"multiple": True} if multiple else {})
        self.serve([reply(tool_calls=[ask("c1", **args)]), reply(content="OK.")], script)
        return self.client.updates("tool_result")[0]["data"]["result"]

    def test_selected_indexes(self):
        self.assertEqual(self.ask_over_the_pipe({"selected": [2]}), 'The user chose: "c".')
        self.assertEqual(self.requests[0]["call_id"], "c1")
        self.assertEqual(self.requests[0]["options"], ["a", "b", "c"])
        self.assertFalse(self.requests[0]["multiple"])

    def test_text_is_read_like_the_terminal(self):
        self.assertEqual(self.ask_over_the_pipe({"text": "2"}), 'The user chose: "b".')

    def test_free_text_and_skip(self):
        self.assertEqual(self.ask_over_the_pipe({"text": "none"}),
                         "The user answered in their own words: none")

    def test_skipped(self):
        self.assertEqual(self.ask_over_the_pipe({"skipped": True}), SKIPPED_NOTE)

    def test_a_malformed_answer_is_asked_again(self):
        result = self.ask_over_the_pipe({"selected": [0, 1]}, {"selected": [5]}, {"selected": [1]})
        self.assertEqual(result, 'The user chose: "b".')
        self.assertIn("exactly one", self.requests[1]["rejected"]["reason"])
        self.assertEqual(self.requests[2]["rejected"]["answer"], {"selected": [5]})

    def test_several_when_multiple(self):
        self.assertEqual(self.ask_over_the_pipe({"selected": [0, 2]}, multiple=True),
                         'The user chose: "a", "c".')

    def test_no_client_answer_means_nobody(self):
        ui = ServeUI(Path("."), None, None)
        ui.request = lambda method, params: None
        self.assertIsNone(ui.ask_user("Q?", [], False))


if __name__ == "__main__":
    unittest.main()
