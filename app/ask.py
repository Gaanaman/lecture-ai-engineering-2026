"""Lecture 1: ask a bare LLM, and report what the answer cost.

    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --temperature 2.0 --repeat 3 "LoRAとは何ですか？"
    uv run python -m app.ask --chat          # keep the conversation history

No tools, no retrieval, one call per question. Ask something specific to this
course and the model either says it does not know or answers plausibly and
wrongly; the correct wording is in data/glossary.json under homework-levels.
Lecture 3 fixes this by giving the model a glossary tool.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# A lost connection or a failed DNS lookup arrives from httpx, not as an APIError.
CALL_ERRORS = (errors.APIError, httpx.TransportError)

# A course-specific question. The correct answer is in data/glossary.json (homework-levels).
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class Reply:
    """One model response, and what it cost to get it."""

    text: str
    model: str
    seconds: float
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    total_tokens: int

    def usage_line(self) -> str:
        parts = [f"input {self.input_tokens}", f"output {self.output_tokens}"]
        if self.thought_tokens:
            parts.append(f"thinking {self.thought_tokens}")
        parts.append(f"total {self.total_tokens}")
        return f"[usage] {' / '.join(parts)} tokens, {self.seconds:.2f}s"


def token_counts(response) -> dict[str, int]:
    """Read the token counts off a response.

    Fields are absent or None depending on the model: thoughts_token_count only
    appears for thinking models. Prefer the API's own total, fall back to the sum
    of the parts, because the API total also covers tool-use tokens.
    """
    usage = getattr(response, "usage_metadata", None)

    def count(name: str) -> int:
        return getattr(usage, name, None) or 0

    counts = {
        "input_tokens": count("prompt_token_count"),
        "output_tokens": count("candidates_token_count"),
        "thought_tokens": count("thoughts_token_count"),
    }
    counts["total_tokens"] = count("total_token_count") or sum(counts.values())
    return counts


def error_message(e: Exception) -> str:
    """One line for the user, whether the API refused the call or the network failed."""
    if isinstance(e, errors.APIError):
        return api_error_message(e.code, e.message)
    return f"Cannot reach the Gemini API ({e}). Check the internet connection and run again."


def generation_config(temperature: float | None) -> types.GenerateContentConfig | None:
    """Only send a config when there is something to set, so the model's default stands."""
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def ask(
    question: str,
    *,
    model: str = MODEL,
    temperature: float | None = None,
    client=None,
) -> Reply:
    """Ask once. Returns the text, the token counts, and the wall-clock time."""
    client = client or genai.Client(api_key=get_api_key())
    started = time.perf_counter()
    response = client.models.generate_content(
        model=model, contents=question, config=generation_config(temperature)
    )
    seconds = time.perf_counter() - started
    return Reply(
        text=response.text, model=model, seconds=seconds, **token_counts(response)
    )


def converse(
    *,
    model: str = MODEL,
    temperature: float | None = None,
    client=None,
    reader=input,
    out=print,
) -> None:
    """Multi-turn mode: follow-up questions that keep the history (implementation level).

    A chat object resends the whole history as input on every turn, so the input
    token count climbs turn by turn while the output count tracks only the new
    answer. Each turn prints both counts and a running total; a long enough
    conversation reaches the model's context window.
    """
    client = client or genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=model, config=generation_config(temperature))

    out(f"[model] {model}   (blank line or Ctrl-D to end)")
    turn = 0
    spent = 0
    while True:
        try:
            question = reader("\n> ").strip()
        except EOFError:
            break
        if not question:
            break

        turn += 1
        started = time.perf_counter()
        try:
            response = chat.send_message(question)
        except CALL_ERRORS as e:
            # One failed turn (rate limit, busy model, network) should not end the conversation.
            out(error_message(e))
            continue
        seconds = time.perf_counter() - started

        reply = Reply(
            text=response.text, model=model, seconds=seconds, **token_counts(response)
        )
        spent += reply.total_tokens
        out(f"\n{reply.text}")
        out(f"[turn {turn}] {reply.usage_line()}  running total {spent} tokens")

    out(f"\n{turn} turns, {spent} tokens in total.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a bare LLM one question")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--model", default=MODEL, help=f"model for this run only (default: {MODEL})"
    )
    parser.add_argument(
        "--temperature", type=float, help="low is steady, high is varied"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="ask the same question N times"
    )
    parser.add_argument(
        "--chat", action="store_true", help="keep the history and ask follow-ups"
    )
    args = parser.parse_args()

    try:
        if args.chat:
            converse(model=args.model, temperature=args.temperature)
            return

        print(f"[model] {args.model}")
        if args.temperature is not None:
            print(f"[temperature] {args.temperature}")
        print(f"[question] {args.question}")

        for run in range(1, args.repeat + 1):
            if args.repeat > 1:
                print(f"\n--- run {run}/{args.repeat}")
            reply = ask(args.question, model=args.model, temperature=args.temperature)
            print(f"\n{reply.text}")
            print(reply.usage_line())
    except CALL_ERRORS as e:
        print(error_message(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
