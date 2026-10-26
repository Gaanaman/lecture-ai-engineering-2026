# AI Engineering Practice 2026: Exercises and Homework

Exercises and homework for AI Engineering Practice 2026 (AIE2026), Matsuo-Iwasawa Laboratory, The University of Tokyo, global track. The repository is a fork of [matsuolab/lecture-ai-engineering-2026](https://github.com/matsuolab/lecture-ai-engineering-2026); the course guide is provided in [README.md](https://github.com/Gaanaman/lecture-ai-engineering-2026/blob/main/README.md).

The codebase is extended lecture by lecture into a learning-support assistant built on the Gemini API.

## Progress

| Lecture | Topic | Homework level | Write-up |
|---|---|---|---|
| 1 | AI engineering and AI-assisted coding | Basic, Implementation, Advanced | [lectures/01/answer.md](https://github.com/Gaanaman/lecture-ai-engineering-2026/blob/main/lectures/01/answer.md) |

## Lecture 1: Principal Results

Two models (`gemini-3.5-flash-lite`, `gemini-3.5-flash`) and three temperature settings were compared, with three runs per setting, on two questions: a general question (what is LoRA?) and a question answerable only from course material (what are the three homework levels?).

1. **Course-specific answers were incorrect in 15 of 15 runs.** Each answer described a plausible beginner-to-advanced sequence of levels that is not present in the course glossary.
2. **Keyword matching proved insufficient as an evaluation.** Eight of the 15 answers were scored as correct by the keyword check, and all eight were incorrect: the expected terms appeared with different meanings.
3. **Conversation history is resent with each turn.** Across three follow-up questions, input rose from 7 to 581 to 1,260 tokens, although each new question comprised a few dozen tokens.
4. **Throughput was limited by response time rather than by the rate limit.** Response times of about 20 s per call were observed on the free tier (2.0 to 2.9 calls per minute, below 10 per minute, the lowest free-tier figure identified in published sources). For `gemini-3.5-flash`, thinking tokens accounted for 55% to 63% of all tokens and are not included in the answer text.

All 21 benchmark calls are recorded in [lectures/01/results/](https://github.com/Gaanaman/lecture-ai-engineering-2026/tree/main/lectures/01/results), and the tables in the write-up are generated from these records.

## Additions to the Codebase

| File | Purpose |
|---|---|
| `app/ask.py` | Single-question client reporting input, output and thinking tokens and response time; options `--temperature`, `--repeat` and `--chat` (follow-up questions with retained history) |
| `app/bench.py` | Model and temperature comparison, paced below the free-tier limit, with retries on HTTP 429 and 503; result tables are written into the write-up |
| `tests/test_ask.py` | 13 offline tests using a fake client; no API key or quota is required |

## Reproduction

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required, together with a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) set as `GEMINI_API_KEY`.

```bash
uv sync --frozen
uv run pytest -q
uv run python -m app.ask --chat
uv run python -m app.bench --write-answer
```

## Scope and Limitations

The results derive from coursework conditions: three runs per setting, a single network connection, and a single day (26 October 2026). They describe these runs only and do not characterise Gemini in general. The free-tier limits in effect were those shown for the project in Google AI Studio.
