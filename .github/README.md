# AI Engineering Practice 2026: exercises and homework

Daniel Adotey's work for **AI Engineering Practice 2026** (AIE2026), the University of Tokyo Matsuo-Iwasawa Lab course, global track. This repository is a fork of [matsuolab/lecture-ai-engineering-2026](https://github.com/matsuolab/lecture-ai-engineering-2026); the course's own guide is in [README.md](https://github.com/Gaanaman/lecture-ai-engineering-2026/blob/main/README.md).

Each lecture adds to the same codebase, a small learning-support assistant built on the Gemini API.

## Progress

| Lecture | Topic | Homework level | Write-up |
|---|---|---|---|
| 1 | AI engineering and AI-assisted coding | Basic, Implementation, Advanced | [lectures/01/answer.md](https://github.com/Gaanaman/lecture-ai-engineering-2026/blob/main/lectures/01/answer.md) |

## Lecture 1 in four results

Two models (`gemini-3.5-flash-lite`, `gemini-3.5-flash`) and three temperatures, three runs per setting, on two questions: a general one (what is LoRA?) and one only this course can answer (what are the three homework levels?).

1. **15 of 15 answers to the course question described homework levels that do not exist.** Each one gave a plausible beginner-to-advanced ladder; none matched the course glossary.
2. **A keyword check is not an evaluation.** It marked 8 of those 15 answers as correct, and all 8 were wrong: the expected words appeared, with the wrong meanings.
3. **Conversation history is resent on every turn.** Over three follow-up questions, input grew from 7 to 581 to 1,260 tokens, while each new question was a few dozen tokens.
4. **Response time, not the rate limit, set the pace.** Calls took about 20 seconds on the free tier (2.0 to 2.9 calls a minute, under 10 a minute, the lowest free-tier limit I found). Flash spent 55% to 63% of its tokens on thinking that is not part of the answer text.

All 21 benchmark calls are recorded in [lectures/01/results/](https://github.com/Gaanaman/lecture-ai-engineering-2026/tree/main/lectures/01/results), and the tables in the write-up are generated from them.

## What I added

| File | Purpose |
|---|---|
| `app/ask.py` | Ask one question; prints input, output and thinking tokens and the response time. `--temperature`, `--repeat`, and `--chat` for follow-up questions with history |
| `app/bench.py` | Runs the model and temperature comparison, paced under the free-tier limit, retries on 429 and 503, and writes the tables into the write-up |
| `tests/test_ask.py` | 13 offline tests with a fake client, so they run without an API key or quota |

## Run it

Python 3.12 and [uv](https://docs.astral.sh/uv/), with a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) in `GEMINI_API_KEY`.

```bash
uv sync --frozen
uv run pytest -q
uv run python -m app.ask --chat
uv run python -m app.bench --write-answer
```

## What this is and isn't

This is coursework: three runs per setting, on one connection, on one day (26 October 2026). The figures describe these runs, not Gemini in general, and the free-tier limits that applied are the ones shown for this project in Google AI Studio.
