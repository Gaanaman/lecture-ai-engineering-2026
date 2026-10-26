# 第2回 解答

The write-up is in English (global track). All four runs were executed on an Apple M1 Max GPU (MPS) rather than a Colab T4. Each notebook is committed with its outputs in `lectures/02/notebooks/`, and its result files are in `lectures/02/results/`.

| Run | Notebook | Level |
|---|---|---|
| Course notebook as written | `notebooks/exercise.ipynb` | Exercise |
| LoRA rank 4 instead of 16 | `notebooks/basic_lora_r4.ipynb` | 基礎 |
| Fictional mobile-money wallet FAQ, in English | `notebooks/implementation_wallet_faq.ipynb` | 実装 |
| Glossary fine-tuning compared with glossary search | `notebooks/advanced_glossary.ipynb` | 発展 |

Changes to the course notebook for the local runs: the GPU check reports the Apple GPU; the install cell is replaced by a check that the pinned library versions (`PINS`) are present; results are written to `lectures/02/results/<run>/`; section 8 (4-bit loading with bitsandbytes, CUDA only) is removed.

## 演習

- 使ったモデル・設定（`result.json` の `config` から）：
  `Qwen/Qwen3.5-0.8B` at revision `2fc06364`, LoRA without quantisation, `lora_r` 16, `lora_alpha` 32, `lr` 2e-4, 20 epochs (40 steps), seed 42, greedy decoding. Libraries: transformers 5.18.0, peft 0.21.1, trl 1.14.2, torch 2.11.0.
- 自分で学習を実行したか（`executed_training_myself`）：はい (all 40 steps, on Apple silicon; 777 s, against 4–5 minutes on a T4).
- 学習前：`format_rate` = 0.095 / `content_rate` = 0.000
- 学習後：`format_rate` = 1.000 / `content_rate` = 0.905
- 退行した質問：none.
- 学習後に残った失敗と、その型：
  Two of the 21 answers failed, and both are of one type: an ungrounded question answered with a neighbouring FAQ item and `needs_review: false`.
  - 「成績はいつ出ますか」 (when are grades released?) was answered with the submission deadline.
  - 「オフィスアワーは事前予約が必要ですか」 (do office hours need booking?) was answered with the office-hour times.
- 知識を問う質問（セクション 6）の結果と気づいたこと：
  Before training, all four answers were format errors. After training, the teacher's name and the next lecture's theme were classified `needs_review`, but the final-assignment deadline, which is not in the FAQ, was asserted with the homework deadline. The homework-deadline control was asserted correctly. (The notebook's 「FAQ にある」 column reads False for the control because it checks only the test questions.)
- 学習で変わったこと・変わらなかったこと：
  The format changed completely. Before training, 19 of the 21 outputs omitted the `needs_review` key (for example `{"answer": "9 月 30 日 23:59"}`); after training, all 21 were valid JSON with both keys. The judgement of what the FAQ does not cover changed only partly: the remaining errors are questions near a covered topic, answered with that topic.

## 宿題

- 取り組んだレベル：基礎・実装・発展 (all three; 発展 selected in the OmniCampus form)

### 基礎: LoRA rank 16 compared with rank 4

Only `lora_r` was changed; the session was restarted and the notebook run from the top.

| `lora_r` | Trainable parameters | Training time | Loss at steps 10 / 20 / 30 / 40 | `format_rate` | `content_rate` |
|---|---|---|---|---|---|
| 16 | 6,389,760 (0.84%) | 777 s | 1.564 / 0.085 / 0.018 / 0.014 | 1.000 | 0.905 |
| 4 | 1,597,440 (0.21%) | 746 s | 1.543 / 0.084 / 0.017 / 0.013 | 1.000 | 0.857 |

A quarter of the trainable parameters reduced training time by 31 s (4%), which suggests that most of the time is spent in the frozen 0.75-billion-parameter model rather than in the LoRA matrices. The final loss was slightly lower at rank 4, yet one more test question failed (three failures against two, all of the same type as in the exercise: 「課題の文字数に制限はありますか」 was answered with the deadline). A lower training loss did not translate into better test answers. The two `result.json` files are in `results/exercise/` and `results/basic_lora_r4/`.

### 実装: a fictional mobile-money wallet FAQ

The five FAQ items cover transfer fees, the daily limit, reversal of a mistaken transfer, PIN reset and cash-out at an agent; all figures are invented. The split into train, valid and test, the JSON format and the `needs_review` design are unchanged. The system prompt and the knowledge questions were translated into English to match the FAQ.

The ungrounded questions are questions a wallet customer plausibly asks that the FAQ does not answer. Several were placed next to a covered topic, so that sharing words with the FAQ does not make them answerable: a fee to receive money (the FAQ covers sending fees), a lost phone (the FAQ covers a forgotten PIN), and reporting a fraudulent agent (the FAQ covers cash-out at an agent).

| | `format_rate` | `content_rate` |
|---|---|---|
| Before training | 1.000 | 0.095 |
| After training (860 s) | 1.000 | 0.857 |

With the English FAQ and prompt, all 21 answers were valid JSON before training, against 2 of 21 with the Japanese course FAQ; the run does not separate the effect of the language from that of the FAQ content. Before training, content was wrong because answers were cut down (`{"answer": "1%"}`) or ungrounded questions were asserted (`{"answer": "No"}` for a monthly fee). After training, three answers failed:

- 'What is the largest fee I can pay for one transfer?' was answered with the reversal item (a wrong FAQ item).
- 'How do I report a fraudulent agent?' (ungrounded) was answered with the reversal item and `needs_review: false`.
- 'At what time does the daily limit start again?' was answered correctly ('reset at midnight'), but the check requires both '5,000' and 'midnight', so it is a scoring failure rather than a model failure.

For the knowledge questions, the customer-support email address moved from asserted to `needs_review`; the two other unknown questions were `needs_review` before and after, and the control was asserted correctly.

### 発展: fine-tuning on the glossary compared with glossary search

**Design.** `scripts/make_sft_dataset.py` turned the 58-term glossary into 116 question-answer records. Six terms (ハーネス, Pi, ツールの名前空間, mcpServers設定, fetchサーバー, ランク) were left out of training, leaving 104 records from 52 terms. The model was trained with LoRA for 8 epochs (104 steps, 2,378 s; loss from 3.30 to 0.096). Twenty-six test questions were answered under four conditions: the base model and the LoRA model, each without and with the glossary search in `app/tools.py`, which the lecture 3 agent uses. An answer counts as correct when it contains both listed phrases from the glossary definition, compared with spaces removed; for a term that does not exist, an answer counts as correct when it declines.

| Group (questions) | Base, no search | Base, with search | LoRA, no search | LoRA, with search |
|---|---|---|---|---|
| Trained term, training wording (6) | 0 | 4 | 6 | 4 |
| Trained term, new wording (6) | 0 | 5 | 2 | 2 |
| Trained term, described without its name (4) | 0 | 0 | 0 | 0 |
| In the glossary, not trained (6) | 0 | 2 | 0 | 1 |
| Not in the glossary (4) | 0 | 3 | 0 | 0 |
| **Total (26)** | **0** | **14** | **8** | **7** |

**Findings.**

1. **Recall after fine-tuning depended on the training wording.** With the training question, all six definitions were reproduced word for word. Reworded, only two of six were correct, and wrong answers mixed in other terms: TestModel was described as 'gemini-3.5-flash-lite', which belongs to 既定モデル.
2. **Fine-tuning gave no knowledge of untrained terms and no ability to decline.** The six held-out terms were answered wrongly, and the four nonexistent terms received confident definitions in the course's own style ('AIE2026の…'), unlike the base model's generic inventions.
3. **Search with the base model scored best (14 of 26).** Three of the four nonexistent terms were declined with 「用語集に記載がありません」. Failures came from two sources: the model summarised the retrieved definition and dropped a listed detail, and the search found nothing.
4. **Name-based search cannot answer a question that does not name the term.** All four described questions failed in every condition; with search, three received 「用語集に記載がありません」 because no entry matched. 'Pi' was also missed, because the search ignores names shorter than three characters.
5. **Fine-tuning reduced the benefit of search in this run.** The LoRA model with search scored 7, against 14 for the base model with search: it declined none of the nonexistent terms and added invented details to retrieved definitions.

**Fine-tuning compared with search.**

- *Fine-tuning* suits a fixed format or behaviour: the exercise reached valid JSON for 21 of 21 questions after 40 steps on 14 examples. It needs no search call and no extra prompt tokens. It does not suit facts: knowledge was limited to the trained wording, untrained or nonexistent terms were answered with invented definitions, a change to the glossary needs retraining (2,378 s here for 52 terms), and the trained model made worse use of retrieved text.
- *Search* reaches any glossary entry that its matching rule finds, without training, can be updated by editing `data/glossary.json`, and gave the only declines for nonexistent terms. Its limits are the matching rule (described questions and short names were missed) and the model's tendency to summarise away detail, at the cost of one search call and the retrieved text in each prompt.
- For course facts that change during the course, search is the better fit; fine-tuning is better kept for output format and behaviour.

**Limitations.**

- Each condition was run once, with 4 to 6 questions per group, on one device; the counts describe these runs only.
- The phrase check is strict: a correct summary that drops one listed detail fails.
- The first training attempt, with the course batch settings (4 per step, accumulation 2), diverged on MPS: the loss rose to 5.57 and then became 0, and the trained model produced only `!`. The rerun used 1 example per step with accumulation 8 (the same effective batch of 8, without padding). The first attempt is kept in `results/advanced_glossary/result_first_attempt_diverged.json`.
- The space removal in the phrase check was added after the first attempt's outputs had been read ('API キー' against 'APIキー'). It raised the base-with-search score of that attempt from 11 to 14 and is applied identically to all four conditions of the rerun.
- Two of the LoRA-with-search answers to nonexistent terms begin 「用語集に該当しません」 (not in the glossary) and continue with invented detail. The decline markers were fixed before the runs and do not include this phrase, so these answers are counted as wrong.

### ファインチューニングが向く場面・向かない場面についての考察

Across the four runs, LoRA fine-tuning changed how the model answers more reliably than what it knows. Format was learned in every run (`format_rate` 1.000 in the exercise, rank-4 and wallet runs). Judgement was not: in each FAQ run, questions close to a covered topic were answered with that topic, and in the glossary run, training removed the ability to decline. Fine-tuning suits stable formats, styles and fixed responses; facts that change, or questions whose correct answer is 'not covered', are better handled by search over a maintained source, with evaluation on questions that the search cannot match.
