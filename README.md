# Retrieval-Augmented QA over Medical Reference Text

A RAG pipeline over *The Gale Encyclopedia of Medicine* (2nd ed., 759 pages)
that answers questions with a source citation attached to every answer, and
can be repointed at a user-uploaded PDF at query time. Built as an
independent project; the retrieval side is evaluated quantitatively below,
not just demoed.

I'm looking for research opportunities in retrieval and applied NLP — this
project, and the evaluation harness in particular, is meant to show how I
approach that work: measure the retriever on its own terms before trusting
what the LLM does with what it's given.

## How it works

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (500 chars, 50 overlap)
     -> all-MiniLM-L6-v2 embeddings -> FAISS index
     -> similarity_search(question, k) -> Mistral-7B-Instruct-v0.3 (RetrievalQA, "stuff" chain)
```

- `memory.py` — builds the FAISS index from PDFs in `data/`.
- `llm.py` — CLI query loop: retrieve top-k chunks, pass them to the LLM with
  a prompt that refuses to answer outside the given context.
- `prac1.py` / `web.py` — Streamlit interfaces, including a mode that accepts
  a user-uploaded PDF and re-indexes against it instead of the encyclopedia.
- `rag_eval.py` — retrieval evaluation harness (below).
- `build_chunk_variants.py` / `rag_eval_compare.py` — build alternate FAISS
  indices with different chunking and compare them on the same questions.

## Evaluation

`rag_eval.py` runs 51 hand-written questions against the actual FAISS index
built by `memory.py` — no LLM call, no mocked retriever. Each question
records the exact PDF page (`source_page`, 0-indexed) it was written from,
and a retrieved chunk counts as relevant if it comes from that page —
**not** if it contains a hand-picked phrase.

That distinction matters and is worth stating explicitly: an earlier version
of this harness judged relevance by whether a chunk *contained* a specific
answer phrase. That ground truth is confounded for a chunking comparison —
change the chunk boundaries and the same source text gets sliced
differently, so whether a phrase survives intact inside one particular
chunk depends on the very thing being varied. Page identity doesn't move
when chunk size changes, so it's the criterion that isolates the retriever
as the only variable.

| k | Recall@k | Precision@k | Hit@1 | MRR |
|---|----------|--------------|-------|-----|
| 3 | 0.863 | 0.490 | 0.725 | 0.794 |
| 5 | 0.863 | 0.392 | 0.725 | 0.794 |

51 questions, real questions from real pages (spanning 47 distinct pages),
not filler.

- **No hybrid comparison yet.** These numbers are all dense-only (MiniLM).
  The obvious next step is a dense+sparse hybrid retriever, run through the
  same 51 questions, the way the ablation on the
  [phonetic search project](../project_1/Phonetic-Similarity-Search-System)
  does for its G2P front-end.

### Chunk-boundary ablation — and why the first version of this result was wrong

The original hypothesis: chunk-boundary misses were the dominant failure
mode, so widening the chunks (and overlap) should raise recall.
`build_chunk_variants.py` builds two alternate indices over the *same* PDF
to test that, and `rag_eval_compare.py` runs the *same* 51 questions
against all three — plus an exact McNemar's test on the paired hit/miss
outcomes, since "recall went up" and "that's a real effect, not noise" are
different questions, and 51 paired outcomes is actually enough to ask the
second one.

| Chunking | Recall@5 | Precision@5 | Hit@1 | MRR |
|---|---|---|---|---|
| 500 chars / 50 overlap (baseline) | 0.863 | 0.392 | **0.725** | **0.794** |
| 1000 chars / 200 overlap | **0.922** | 0.380 | 0.647 | 0.761 |
| NLTK sentence-aware, ~1000 chars | 0.902 | 0.361 | 0.667 | 0.761 |

McNemar's test, baseline vs. 1000/200 (3 discordant pairs: baseline missed
and wide hit on all 3, baseline never hit where wide missed): **p = 0.25,
not significant at α = 0.05.** Baseline vs. sentence-aware (2 discordant
pairs): **p = 0.50, not significant.**

**This is a materially different result from an earlier pass of this
ablation**, which used the answer-phrase ground truth described above and
reported a 22-question Recall@5 jump from 0.64 to 0.77. That earlier number
was real in the sense that it was computed, not invented — but it was
measuring two things at once (the retriever, and how cleanly a fixed phrase
happened to survive re-chunking), and 22 questions with only a handful of
flips isn't enough to call the difference anything more than "went the
right direction once." With the deconfounded ground truth and more than
double the questions:

- Recall@5 did move up for wider chunking (0.86 -> 0.92) — same direction
  as before, which is mildly reassuring — but the gap rests on 3 discordant
  questions out of 51, and McNemar's test says that's consistent with
  noise.
- **Hit@1 and MRR moved the other way** (0.73 -> 0.65 Hit@1 for
  1000/200). Wider chunks bring in the right page more often but rank it
  first *less* often, plausibly because a wider window pulls in more
  competing, tangentially-related chunks from the same or nearby pages.
  That's the opposite of what the "improved Recall@5" headline implies
  about the system getting straightforwardly better.

Net: wider chunks preserving more context is a well-founded expectation,
and the data doesn't contradict it — but it doesn't clear the bar to state
it as a demonstrated result, either. That's a real, useful distinction:
*plausible* and *demonstrated* are different claims, and the gap between
them is most of what separates a defensible research line from a
misleading one. The honest version of this line, right now, is "wider
chunking moved recall in the expected direction on 51 questions, but the
effect isn't statistically distinguishable from noise, and precision-side
metrics moved the other way" — not "improved Recall@5 by 21%."

## Other interfaces

The Streamlit app (`prac1.py`) also exposes non-RAG chat personas (CodeBot,
ChatMate, DocuBot, IdeaSpark, TechBot) built on the same LLM call for
general conversation, coding help, and document rewriting. They're not part
of the evaluation above — TOMY (the encyclopedia bot) is the one actually
being measured.

## Setup

```bash
pip install -r requirements.txt
# .env: HF_TOKEN=your_huggingface_api_token
python memory.py              # build the baseline (500/50) FAISS index from data/
python rag_eval.py            # run the retrieval evaluation, 51 questions (no HF_TOKEN needed)
python build_chunk_variants.py  # build the 1000/200 and sentence-aware indices
python rag_eval_compare.py    # compare all three chunking strategies
streamlit run prac1.py        # full app (needs HF_TOKEN for the LLM)
```

`rag_eval.py` only needs the embedding model (local, via
`sentence-transformers`) — it doesn't call the Hugging Face endpoint, so it
runs without an API token.

## Stack

Python · LangChain · FAISS · Hugging Face (`all-MiniLM-L6-v2`,
`Mistral-7B-Instruct-v0.3`) · Streamlit · PyPDF
