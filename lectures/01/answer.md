# 第1回 解答

The write-up is in English (global track). Code produced for this lecture:

- `app/ask.py`: reports input, output and thinking token counts and the response time; adds `--temperature`, `--repeat` and a multi-turn `--chat` mode
- `app/bench.py`: runs the same questions across models and temperatures and writes the tables in this file
- `tests/test_ask.py`: offline tests for both, using a fake client, so that no API key or quota is required

## 演習

- 講座に固有の質問（宿題の3レベル）へのモデルの回答（要約でよい）：
  The `gemini-3.5-flash-lite` output presented the three levels as a probable difficulty sequence, 初級・基礎 (beginner), 中級・応用 (intermediate) and 上級・発展 (advanced), with a detailed description of each level, a statement that the structure was a likely guess, and a request for the course text.
- 用語集の正しい意味と比べて気づいたこと：
  The glossary entry `homework-levels` defines three levels, 基礎 (review of the lecture), 実装 (writing one's own code) and 発展 (investigation and trials), and states that any one of them completes the week. In the output, 基礎 and 発展 appear only as labels for easier and harder work, 中級 (intermediate) occupies the position of 実装 (implementation), the levels are described as difficulty steps rather than review, own code and investigation, and the completion rule is absent. The output is hedged (「可能性が高いです」) but still presents a detailed structure that does not exist in the course: a hallucination accompanied by a hedge.
- 表示できたトークン数（入力 / 出力）：
  LoRA question `input 7 / output 697 / total 704 tokens, 18.46s`; course question `input 21 / output 434 / total 455 tokens, 17.89s`.

## 宿題

- 取り組んだレベル：基礎・実装・発展 (all three; 発展 selected in the OmniCampus form)

### やったこと

**基礎.** Two questions were selected deliberately. "LoRAとは何ですか？" concerns a general term covered by pre-training data; the homework-levels question is specific to this course, and its answer is therefore absent from the pre-training data. Each setting was run three times. Two models were compared at their default temperature, and one model at temperatures 0.0, 1.0 and 2.0. Two measures were used that do not depend on a reader's judgement:

- `keywords`: whether the answer contains every expected term (基礎, 実装 and 発展 for the course question)
- `distinct`: how many of the three answers differ from each other word for word

Google's Gemini 3 guide recommends leaving temperature at its default of 1.0 and warns that lower values can cause looping or weaker reasoning ([Gemini 3 Developer Guide](https://ai.google.dev/gemini-api/docs/gemini-3)). The 0.0 and 2.0 runs were included to measure the effect of the setting, not as a recommendation to use it.

**実装.** A multi-turn mode, `uv run python -m app.ask --chat`, retains the conversation history. `client.chats.create()` returns a chat object that stores each turn and resends the full history with every new question. A usage line is printed for each turn, so the cost of the history is visible as input tokens rising turn by turn. A rate-limit or network error on one turn is reported in a single line, and the conversation continues.

**発展.** In `app/bench.py`, each call is timed with `time.perf_counter()` around the API request, so the figure includes network time as well as generation time. Token counts are read from `response.usage_metadata` (`prompt_token_count`, `candidates_token_count`, `thoughts_token_count`, `total_token_count`). To remain within the free tier, calls are started at least 6 seconds apart (at most 10 requests per minute). After a 429 the call is retried after 60 seconds, and after a 503 or a dropped connection after 10 seconds, for up to three attempts; the run is stopped if the API remains unreachable. Raw results for every call are stored in `lectures/01/results/`, so each table row can be traced to the calls behind it.

### 結果

Models, default temperature (`uv run python -m app.bench --write-answer`):

<!-- bench:models:start -->

| model | temp | question | runs | mean s | min–max s | in | out | thinking | total | keywords | distinct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash` | default | course | 3 | 20.04 | 14.79–26.45 | 21 | 653 | 1148 | 1822 | 1/3 | 3/3 |
| `gemini-3.5-flash-lite` | default | course | 3 | 20.22 | 18.39–21.31 | 21 | 591 | 0 | 612 | 2/3 | 3/3 |
| `gemini-3.5-flash` | default | general | 3 | 39.76 | 17.50–69.06 | 7 | 829 | 1035 | 1871 | 3/3 | 3/3 |
| `gemini-3.5-flash-lite` | default | general | 3 | 20.49 | 19.29–21.39 | 7 | 703 | 0 | 710 | 3/3 | 3/3 |

`keywords` counts the runs whose answer contained every expected term; `distinct` counts how many of the runs differed from each other word for word.

| model | calls | tokens | mean s | tokens/s | serial calls per minute |
|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | 6 | 3965 | 20.36 | 33 | 2.9 |
| `gemini-3.5-flash` | 6 | 11080 | 29.90 | 80 | 2.0 |

12 calls, 0 failed, 15045 tokens in total. Raw results: `lectures/01/results/bench-20261026-170227.json`.

<!-- bench:models:end -->

Temperature, `gemini-3.5-flash-lite`, course question (`uv run python -m app.bench --models gemini-3.5-flash-lite --questions course --temperatures 0.0 1.0 2.0 --write-answer temperature`):

<!-- bench:temperature:start -->

| model | temp | question | runs | mean s | min–max s | in | out | thinking | total | keywords | distinct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | 0.0 | course | 3 | 30.88 | 15.41–48.60 | 21 | 595 | 0 | 616 | 2/3 | 3/3 |
| `gemini-3.5-flash-lite` | 1.0 | course | 3 | 20.15 | 17.64–23.87 | 21 | 593 | 0 | 614 | 2/3 | 3/3 |
| `gemini-3.5-flash-lite` | 2.0 | course | 3 | 17.39 | 14.77–20.52 | 21 | 554 | 0 | 575 | 1/3 | 3/3 |

`keywords` counts the runs whose answer contained every expected term; `distinct` counts how many of the runs differed from each other word for word.

| model | calls | tokens | mean s | tokens/s | serial calls per minute |
|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | 9 | 5416 | 22.81 | 30 | 2.6 |

9 calls, 0 failed, 5416 tokens in total. Raw results: `lectures/01/results/bench-20261026-170553.json`.

<!-- bench:temperature:end -->

Conversation history, `gemini-3.5-flash-lite` (`uv run python -m app.bench --chat-demo --write-answer chat`):

<!-- bench:chat:start -->

Questions, in order:

1. LoRAとは何ですか？
2. QLoRAとはどう違いますか？
3. 無料のT4 GPUで学習するなら、その2つのどちらが向いていますか？

```text
[turn 1] [usage] input 7 / output 564 / total 571 tokens, 15.86s  running total 571 tokens
[turn 2] [usage] input 581 / output 658 / total 1239 tokens, 29.47s  running total 1810 tokens
[turn 3] [usage] input 1260 / output 574 / total 1834 tokens, 15.63s  running total 3644 tokens
```

<!-- bench:chat:end -->

### Rate limits on the free tier

The Gemini API limits each **project** (not each key) on three measures: requests per minute (RPM), input tokens per minute (TPM) and requests per day (RPD). Exceeding any one of them returns HTTP 429, and the daily count resets at midnight Pacific time ([Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)). The current figures for a project are shown in Google AI Studio.

- **With short prompts, the request limit is reached before the token limit.** Each question here comprises a few dozen input tokens (the `in` column), so a per-minute token limit would require hundreds or thousands of calls per minute to be reached, far more than the per-minute request limit permits. The last column of the second table is 60 divided by the mean response time, that is, the number of calls an unpaced loop would make per minute; where this figure exceeds the project's RPM limit, 429 errors are returned. In these runs it was 2.9 calls per minute for Flash-Lite and 2.0 for Flash, below 10 RPM, the lowest free-tier figure identified in published sources. Each benchmark call took at least 14.77 s, longer than the 6-second spacing, so no pacing delay was applied: on this connection, throughput was limited by response time rather than by the rate limit. The single error was of a different kind: one Flash call returned 503 (model busy) and succeeded on retry 10 seconds later, which indicates that capacity errors occur on the free tier in addition to rate limits.
- **Thinking tokens add time but are not shown.** They are counted in `total_token_count`, are not part of `response.text`, and the time spent producing them is included in the measured response time. Flash produced 967–1,281 thinking tokens per call, more than its visible answer (481–885 tokens): on average 55% of all tokens on the general question (1,035 of 1,871) and 63% on the course question (1,148 of 1,822). Flash-Lite produced none. Flash was slower on the general question (mean 39.76 s against 20.49 s, including one 69.06 s call) but not on the course question (20.04 s against 20.22 s); in these runs, therefore, thinking tokens did not add a consistent delay.
- **History raises the input count on each turn.** In `--chat`, each follow-up resends the earlier turns, so input tokens per minute grow with the length of the conversation rather than with the length of the new question. In the run, input rose from 7 to 581 to 1,260 tokens over three turns, while each answer comprised 564–658 tokens: the third question was about 21 tokens, and the remainder of its input consisted of the first two questions and answers sent again. The third answer compared LoRA and QLoRA, which shows that その2つ was resolved from the history.
- **An agent makes several calls per question.** From lecture 3, the agent calls the model more than once per question (to select a tool, then to answer). With k calls per question, the RPM limit permits RPM / k questions per minute, which explains the course README's warning that agents reach the limit sooner than `app/ask.py`.

### 気づき・疑問

- **The two questions differed in content, not in the degree of wording variation.** Each of the seven cells produced three differently worded answers (`distinct` 3/3). The six general answers all contained the expected terms (LoRA, low-rank, fine-tuning). The fifteen course answers each presented an invented sequence of levels: the level-2 name was 応用 (applied) in 12 of them, and the course's 実装 appeared as a level name once (Flash, run 1, alongside 応用).
- **Temperature changed the wording, not the correctness.** Temperature 0.0 did not make the output repeatable: the three answers differed at each setting. Mean output length varied little (595, 593 and 554 tokens at 0.0, 1.0 and 2.0), and no 0.0 run showed the looping described in Google's guide (outputs of 558–621 tokens). One 48.60 s call raised the 0.0 mean to 30.88 s; the other two took 28.62 s and 15.41 s. The answers were incorrect at all three settings.
- **The keyword check failed as a test of correctness.** It scored 8 of the 15 course answers as hits, and all 8 are false positives: the words 基礎, 実装 and 発展 appear in passing or as difficulty labels. A stronger check would compare each answer with the glossary definition, by a human rater or by a second model acting as a judge, which is the kind of evaluation covered later in the course.
- `distinct` counts any wording change as a different answer and therefore cannot separate a reworded answer from a substantively different one.
