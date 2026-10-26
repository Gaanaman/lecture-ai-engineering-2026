"""Tests for app.ask and app.bench, with a fake client (no API key, no network)."""

from dataclasses import dataclass

import httpx

from app import ask, bench


@dataclass
class Usage:
    prompt_token_count: int | None
    candidates_token_count: int | None
    thoughts_token_count: int | None = None
    total_token_count: int | None = None


class Response:
    def __init__(self, text: str, usage: Usage):
        self.text = text
        self.usage_metadata = usage


class Models:
    """Stands in for client.models; answers with a function of the call."""

    def __init__(self, answer=lambda model, question, n: f"{model}: answer {n}"):
        self.answer = answer
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        text = self.answer(model, contents, len(self.calls))
        return Response(text, Usage(len(contents), len(text), 0, None))


class Chat:
    """Resends the whole history on every turn, as the real chat object does."""

    def __init__(self):
        self.history: list[str] = []

    def send_message(self, message: str):
        self.history.append(message)
        text = f"reply to {message}"
        sent = sum(len(m) for m in self.history)
        self.history.append(text)
        return Response(text, Usage(sent, len(text), 0, None))


class Offline(Models):
    """Every call fails the way a dropped connection does."""

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        raise httpx.ConnectError("nodename nor servname provided, or not known")


class DropsFirstTurn(Chat):
    def __init__(self):
        super().__init__()
        self.dropped = False

    def send_message(self, message: str):
        if not self.dropped:
            self.dropped = True
            raise httpx.ConnectError("connection dropped")
        return super().send_message(message)


class OfflineChat(Chat):
    def send_message(self, message: str):
        raise httpx.ConnectError("connection dropped")


class Chats:
    def __init__(self, chat=Chat):
        self.chat = chat
        self.created = []

    def create(self, *, model, config=None):
        self.created.append({"model": model, "config": config})
        return self.chat()


class Client:
    def __init__(self, models: Models | None = None, chat=Chat):
        self.models = models or Models()
        self.chats = Chats(chat)


def test_token_counts_reads_the_api_total_when_present():
    response = Response("x", Usage(10, 20, 5, 40))
    assert ask.token_counts(response) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "thought_tokens": 5,
        "total_tokens": 40,
    }


def test_token_counts_sums_the_parts_when_the_total_is_missing():
    response = Response("x", Usage(10, None, None, None))
    counts = ask.token_counts(response)
    assert counts["output_tokens"] == 0
    assert counts["total_tokens"] == 10


def test_ask_reports_tokens_time_and_model():
    reply = ask.ask("What is LoRA?", model="m1", client=Client())
    assert reply.model == "m1"
    assert reply.input_tokens == len("What is LoRA?")
    assert reply.seconds >= 0
    assert "input" in reply.usage_line() and "output" in reply.usage_line()


def test_temperature_is_sent_only_when_given():
    client = Client()
    ask.ask("q", model="m", client=client)
    ask.ask("q", model="m", temperature=1.5, client=client)
    first, second = client.models.calls
    assert first["config"] is None
    assert second["config"].temperature == 1.5


def test_converse_keeps_history_so_input_tokens_grow():
    printed: list[str] = []
    questions = iter(["What is LoRA?", "And QLoRA?", "Which one fits a T4?", ""])
    ask.converse(
        model="m", client=Client(), reader=lambda _prompt: next(questions), out=printed.append
    )
    turns = [line for line in printed if line.startswith("[turn ")]
    assert len(turns) == 3
    inputs = [int(line.split("input ")[1].split(" ")[0]) for line in turns]
    assert inputs == sorted(inputs) and inputs[0] < inputs[-1]
    assert printed[-1].startswith("\n3 turns")


def test_converse_reports_a_network_error_and_carries_on():
    printed: list[str] = []
    questions = iter(["first", "second", ""])
    ask.converse(
        model="m",
        client=Client(chat=DropsFirstTurn),
        reader=lambda _prompt: next(questions),
        out=printed.append,
    )
    assert any("Cannot reach the Gemini API" in line for line in printed)
    assert sum(line.startswith("[turn ") for line in printed) == 1


def test_keyword_hit_needs_one_word_from_every_group():
    expect = bench.QUESTIONS["course"]["expect"]
    assert bench.keyword_hit("基礎・実装・発展の3段階", expect)
    assert not bench.keyword_hit("初級・中級・上級の3段階", expect)


def test_run_grid_makes_one_call_per_cell():
    client = Client(Models(answer=lambda model, q, n: "基礎 実装 発展"))
    rows = bench.run_grid(
        models=["m1", "m2"],
        questions=["general", "course"],
        temperatures=[None, 0.0],
        repeat=2,
        min_interval=0,
        client=client,
        out=lambda *_: None,
    )
    assert len(rows) == 2 * 2 * 2 * 2 == len(client.models.calls)
    assert all(r.keyword_hit for r in rows if r.question == "course")
    summary = bench.summarise(rows)
    assert len(summary) == 8
    assert all(s["runs"] == 2 and s["distinct_answers"] == 1 for s in summary)


def test_grid_stops_at_the_first_network_failure():
    client = Client(Offline())
    rows = bench.run_grid(
        models=["m1", "m2"],
        questions=["general", "course"],
        temperatures=[None],
        repeat=3,
        min_interval=0,
        attempts=1,
        client=client,
        out=lambda *_: None,
    )
    assert len(rows) == 1 == len(client.models.calls)
    assert rows[0].unreachable and "Cannot reach the Gemini API" in rows[0].error


def test_chat_demo_returns_none_when_no_turn_succeeds():
    assert bench.chat_demo("m", client=Client(chat=OfflineChat), out=lambda *_: None) is None


def test_chat_demo_reports_one_usage_line_per_follow_up():
    block = bench.chat_demo("m", client=Client(), out=lambda *_: None)
    assert block.count("[turn ") == len(bench.FOLLOW_UPS)
    assert bench.FOLLOW_UPS[1] in block


def test_write_answer_replaces_only_its_own_block(tmp_path):
    models_start, models_end = bench.markers("models")
    temp_start, temp_end = bench.markers("temperature")
    answer = tmp_path / "answer.md"
    answer.write_text(
        f"before\n{models_start}\nold\n{models_end}\nmiddle\n"
        f"{temp_start}\nkeep\n{temp_end}\nafter\n",
        encoding="utf-8",
    )
    assert bench.write_answer("new table", "models", answer)
    text = answer.read_text(encoding="utf-8")
    assert "old" not in text and "new table" in text and "keep" in text
    assert text.startswith("before") and "middle" in text and text.rstrip().endswith("after")


def test_write_answer_leaves_a_file_without_markers_alone(tmp_path):
    answer = tmp_path / "answer.md"
    answer.write_text("no markers here\n", encoding="utf-8")
    assert not bench.write_answer("table", "models", answer)
    assert answer.read_text(encoding="utf-8") == "no markers here\n"
