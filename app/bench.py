"""Lecture 1 homework: run the same questions across models and temperatures, and measure.

    uv run python -m app.bench                                  # 2 models x 2 questions x 3 runs
    uv run python -m app.bench --write-answer                   # tables into answer.md (models block)
    uv run python -m app.bench --models gemini-3.5-flash-lite --questions course \
        --temperatures 0.0 1.0 2.0 --write-answer temperature   # second block
    uv run python -m app.bench --chat-demo --write-answer chat  # history cost, third block

One cell of the grid is one API call, so a run makes
models x questions x temperatures x repeat calls. --min-interval spaces the calls
out so a free-tier key stays under its per-minute request limit; a 429 is retried
after a wait rather than treated as a failure.

Raw results are written to lectures/01/results/ as JSON, so the tables in
answer.md can be traced back to the calls that produced them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors

from app.ask import (
    CALL_ERRORS,
    DEFAULT_QUESTION,
    converse,
    error_message,
    generation_config,
    token_counts,
)
from app.config import MODEL, get_api_key

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "lectures" / "01" / "results"
ANSWER_FILE = REPO_ROOT / "lectures" / "01" / "answer.md"
DEFAULT_SECTION = "models"


def markers(section: str) -> tuple[str, str]:
    return f"<!-- bench:{section}:start -->", f"<!-- bench:{section}:end -->"

# Two questions that sit on opposite sides of what pre-training covers.
# `expect` is a keyword rubric: a run counts as a hit only if at least one
# alternative from every group appears in the answer. It does not judge whether
# the answer is good, only whether the expected content is present at all.
QUESTIONS: dict[str, dict] = {
    "general": {
        "text": "LoRAとは何ですか？",
        "note": "general term, well covered by pre-training",
        "expect": [
            ["LoRA", "Low-Rank Adaptation"],
            ["低ランク", "Low-Rank", "low-rank"],
            ["ファインチューニング", "微調整", "fine-tun"],
        ],
    },
    "course": {
        "text": DEFAULT_QUESTION,
        "note": "specific to AIE2026, absent from pre-training",
        "expect": [["基礎"], ["実装"], ["発展"]],
    },
}

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]

# Follow-ups that need the history: question 2 leaves its subject (LoRA) unstated,
# and question 3 refers to the two methods as その2つ.
FOLLOW_UPS = [
    "LoRAとは何ですか？",
    "QLoRAとはどう違いますか？",
    "無料のT4 GPUで学習するなら、その2つのどちらが向いていますか？",
]


@dataclass
class Row:
    """One API call."""

    model: str
    question: str
    temperature: float | None
    run: int
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    characters: int = 0
    keyword_hit: bool = False
    text: str = ""
    error: str = ""
    unreachable: bool = False


@dataclass
class Pacer:
    """Keeps at least `interval` seconds between the start of one call and the next."""

    interval: float
    _last: float = field(default=0.0, repr=False)

    def wait(self) -> None:
        gap = self.interval - (time.monotonic() - self._last)
        if self._last and gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


def keyword_hit(text: str, expect: list[list[str]]) -> bool:
    return all(any(word in text for word in group) for group in expect)


def call_once(client, model: str, question: str, temperature: float | None):
    return client.models.generate_content(
        model=model, contents=question, config=generation_config(temperature)
    )


def run_cell(
    client, model: str, key: str, temperature: float | None, run: int, *, attempts: int, out
) -> Row:
    """One call, retried on a rate limit, a busy model or a dropped connection."""
    question = QUESTIONS[key]
    row = Row(model=model, question=key, temperature=temperature, run=run)

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            response = call_once(client, model, question["text"], temperature)
        except CALL_ERRORS as e:
            row.error = error_message(e)
            row.unreachable = not isinstance(e, errors.APIError)
            if attempt == attempts:
                out(f"    gave up after {attempts} attempts: {row.error}")
                return row
            wait = 60 if getattr(e, "code", None) == 429 else 10
            out(f"    {row.error} waiting {wait}s, attempt {attempt + 1}/{attempts}")
            time.sleep(wait)
            continue

        row.seconds = time.perf_counter() - started
        row.text = response.text or ""
        row.characters = len(row.text)
        row.keyword_hit = keyword_hit(row.text, question["expect"])
        for name, value in token_counts(response).items():
            setattr(row, name, value)
        row.error = ""
        row.unreachable = False
        return row

    return row


def run_grid(
    *,
    models: list[str],
    questions: list[str],
    temperatures: list[float | None],
    repeat: int,
    min_interval: float,
    attempts: int = 3,
    client=None,
    out=print,
) -> list[Row]:
    client = client or genai.Client(api_key=get_api_key())
    pacer = Pacer(min_interval)
    rows: list[Row] = []

    total = len(models) * len(questions) * len(temperatures) * repeat
    out(f"{total} calls, at least {min_interval:g}s apart\n")

    for model, key, temperature in itertools.product(models, questions, temperatures):
        out(f"  {model} / {key} / temperature {_temp(temperature)}")
        for run in range(1, repeat + 1):
            pacer.wait()
            row = run_cell(client, model, key, temperature, run, attempts=attempts, out=out)
            rows.append(row)
            if row.unreachable:
                # With no connection, every remaining call would fail the same way.
                out("  stopped: the Gemini API is unreachable")
                return rows
            if not row.error:
                out(
                    f"    run {run}: {row.seconds:5.2f}s  "
                    f"{row.total_tokens:5d} tokens  "
                    f"{row.characters:5d} chars  "
                    f"keywords {'yes' if row.keyword_hit else 'no'}"
                )
    return rows


def summarise(rows: list[Row]) -> list[dict]:
    """Collapse the runs of each cell into one line."""
    cells: dict[tuple, list[Row]] = {}
    for row in rows:
        cells.setdefault((row.model, row.question, row.temperature), []).append(row)

    summary = []
    for (model, question, temperature), group in cells.items():
        done = [r for r in group if not r.error]
        if not done:
            summary.append(
                {
                    "model": model,
                    "question": question,
                    "temperature": temperature,
                    "runs": 0,
                    "failed": len(group),
                    "error": group[0].error,
                }
            )
            continue
        summary.append(
            {
                "model": model,
                "question": question,
                "temperature": temperature,
                "runs": len(done),
                "failed": len(group) - len(done),
                "mean_seconds": statistics.fmean(r.seconds for r in done),
                "min_seconds": min(r.seconds for r in done),
                "max_seconds": max(r.seconds for r in done),
                "mean_input_tokens": statistics.fmean(r.input_tokens for r in done),
                "mean_output_tokens": statistics.fmean(r.output_tokens for r in done),
                "mean_thought_tokens": statistics.fmean(r.thought_tokens for r in done),
                "mean_total_tokens": statistics.fmean(r.total_tokens for r in done),
                "mean_characters": statistics.fmean(r.characters for r in done),
                "keyword_hits": sum(r.keyword_hit for r in done),
                "distinct_answers": len({r.text for r in done}),
                "error": "",
            }
        )
    return summary


def per_model(rows: list[Row]) -> list[dict]:
    """Throughput per model, and what it implies for a per-minute request limit."""
    models: dict[str, list[Row]] = {}
    for row in rows:
        if not row.error:
            models.setdefault(row.model, []).append(row)

    out = []
    for model, group in models.items():
        seconds = statistics.fmean(r.seconds for r in group)
        tokens = sum(r.total_tokens for r in group)
        out.append(
            {
                "model": model,
                "calls": len(group),
                "total_tokens": tokens,
                "mean_seconds": seconds,
                "tokens_per_second": statistics.fmean(
                    r.total_tokens / r.seconds for r in group if r.seconds > 0
                ),
                "calls_per_minute_if_unpaced": 60 / seconds if seconds else 0.0,
            }
        )
    return out


def _temp(value: float | None) -> str:
    return "default" if value is None else f"{value:.1f}"


def markdown(rows: list[Row], summary: list[dict], models: list[dict], raw_path: Path) -> str:
    lines = [
        "| model | temp | question | runs | mean s | min–max s | in | out | thinking | total | keywords | distinct |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(summary, key=lambda s: (s["question"], s["model"], str(s["temperature"]))):
        if not s["runs"]:
            lines.append(
                f"| `{s['model']}` | {_temp(s['temperature'])} | {s['question']} | 0 | "
                f"- | - | - | - | - | - | - | {s['error']} |"
            )
            continue
        lines.append(
            f"| `{s['model']}` | {_temp(s['temperature'])} | {s['question']} | {s['runs']} | "
            f"{s['mean_seconds']:.2f} | {s['min_seconds']:.2f}–{s['max_seconds']:.2f} | "
            f"{s['mean_input_tokens']:.0f} | {s['mean_output_tokens']:.0f} | "
            f"{s['mean_thought_tokens']:.0f} | "
            f"{s['mean_total_tokens']:.0f} | {s['keyword_hits']}/{s['runs']} | "
            f"{s['distinct_answers']}/{s['runs']} |"
        )

    lines += [
        "",
        "`keywords` counts the runs whose answer contained every expected term; "
        "`distinct` counts how many of the runs differed from each other word for word.",
        "",
        "| model | calls | tokens | mean s | tokens/s | serial calls per minute |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        lines.append(
            f"| `{m['model']}` | {m['calls']} | {m['total_tokens']} | "
            f"{m['mean_seconds']:.2f} | {m['tokens_per_second']:.0f} | "
            f"{m['calls_per_minute_if_unpaced']:.1f} |"
        )

    failed = sum(1 for r in rows if r.error)
    lines += [
        "",
        f"{len(rows)} calls, {failed} failed, "
        f"{sum(r.total_tokens for r in rows)} tokens in total. "
        f"Raw results: `{raw_path.relative_to(REPO_ROOT)}`.",
    ]
    return "\n".join(lines)


def chat_demo(model: str, client=None, out=print) -> str | None:
    """Run the scripted follow-ups through ask.converse and keep the usage lines.

    Returns None when no turn succeeded, so a failed run leaves answer.md alone.
    """
    lines: list[str] = []
    questions = iter([*FOLLOW_UPS, ""])

    def record(line: str) -> None:
        lines.append(line)
        out(line)

    converse(model=model, client=client, reader=lambda _prompt: next(questions), out=record)
    usage = [line for line in lines if line.startswith("[turn ")]
    if not usage:
        return None
    asked = "\n".join(f"{n}. {q}" for n, q in enumerate(FOLLOW_UPS, 1))
    return f"Questions, in order:\n\n{asked}\n\n```text\n" + "\n".join(usage) + "\n```"


def write_answer(table: str, section: str = DEFAULT_SECTION, path: Path | None = None) -> bool:
    """Replace whatever sits between one pair of markers in answer.md with the new tables.

    Each block has its own markers, so the model run and the temperature run can
    both live in answer.md without overwriting each other.
    """
    path = path or ANSWER_FILE
    start, end = markers(section)
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n\n{table}\n\n{end}{tail}", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models, temperatures and questions")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--questions", nargs="+", choices=list(QUESTIONS), default=list(QUESTIONS))
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        help="omit to use each model's own default",
    )
    parser.add_argument("--repeat", type=int, default=3, help="runs per cell")
    parser.add_argument(
        "--chat-demo",
        action="store_true",
        help="instead of the grid, run three follow-up questions with history on the first model",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=6.0,
        help="seconds between calls; 6s is 10 requests per minute",
    )
    parser.add_argument(
        "--write-answer",
        nargs="?",
        const=DEFAULT_SECTION,
        metavar="SECTION",
        help=f"put the tables into that block of lectures/01/answer.md (default: {DEFAULT_SECTION})",
    )
    args = parser.parse_args()

    if args.chat_demo:
        block = chat_demo(args.models[0])
        if block is None:
            print("no turn succeeded, so answer.md is unchanged")
        elif args.write_answer:
            _report_write(block, args.write_answer)
        return

    temperatures: list[float | None] = args.temperatures if args.temperatures else [None]
    started = time.perf_counter()
    rows = run_grid(
        models=args.models,
        questions=args.questions,
        temperatures=temperatures,
        repeat=args.repeat,
        min_interval=args.min_interval,
    )
    elapsed = time.perf_counter() - started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    raw_path = RESULTS_DIR / f"bench-{stamp}.json"
    raw_path.write_text(
        json.dumps(
            {
                "generated_utc": stamp,
                "default_model": MODEL,
                "wall_clock_seconds": elapsed,
                "rows": [asdict(r) for r in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table = markdown(rows, summarise(rows), per_model(rows), raw_path)
    print(f"\n{table}\n")
    print(f"wall clock {elapsed:.1f}s, raw results in {raw_path}")

    if args.write_answer:
        if any(not r.error for r in rows):
            _report_write(table, args.write_answer)
        else:
            print("no call succeeded, so answer.md is unchanged")


def _report_write(block: str, section: str) -> None:
    if write_answer(block, section):
        print(f"updated the {section} block of {ANSWER_FILE.relative_to(REPO_ROOT)}")
    else:
        start, end = markers(section)
        print(f"{start} / {end} not found in answer.md; nothing written")


if __name__ == "__main__":
    main()
