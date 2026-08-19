"""
Runs the same 51-question TEST_SET from rag_eval.py (page-identity ground
truth) against three FAISS indices built with different chunking
strategies, reports the numbers side by side, and runs McNemar's test on
the paired hit/miss outcomes between the baseline and each alternative —
because "did recall go up" and "is that a real effect, not noise" are
different questions, and with 51 paired binary outcomes this one is
actually answerable.

  baseline  - vectorstore/db_faiss                  (500 chars / 50 overlap)
  wide      - vectorstore/db_faiss_wide_1000_200     (1000 chars / 200 overlap)
  sentence  - vectorstore/db_faiss_sentence_1000     (NLTK sentence-aware, ~1000 chars)

Run after build_chunk_variants.py has built the two new indices:
  python rag_eval_compare.py
"""

import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from scipy.stats import binomtest

from rag_eval import TEST_SET, evaluate
import rag_eval

VARIANTS = {
    "baseline_500_50": "vectorstore/db_faiss",
    "wide_1000_200": "vectorstore/db_faiss_wide_1000_200",
    "sentence_1000": "vectorstore/db_faiss_sentence_1000",
}


def mcnemar(hits_a: dict, hits_b: dict, name_a: str, name_b: str):
    """Exact McNemar's test (binomial on discordant pairs) for two paired
    binary classifiers evaluated on the same questions."""
    questions = list(hits_a.keys())
    b = sum(1 for q in questions if hits_a[q] and not hits_b[q])  # a hit, b miss
    c = sum(1 for q in questions if not hits_a[q] and hits_b[q])  # a miss, b hit
    discordant = b + c

    print(f"\n{name_a} vs {name_b} — McNemar's test on paired hit/miss (k=5)")
    print(f"  {name_a} hit, {name_b} miss: {b}")
    print(f"  {name_a} miss, {name_b} hit: {c}")
    print(f"  discordant pairs: {discordant}")

    if discordant == 0:
        print("  No discordant pairs — identical hit/miss pattern, nothing to test.")
        return {"b": b, "c": c, "discordant": 0, "p_value": None}

    result = binomtest(min(b, c), discordant, 0.5, alternative="two-sided")
    p = result.pvalue
    print(f"  exact McNemar p-value: {p:.4f}"
          + ("  (significant at alpha=0.05)" if p < 0.05 else "  (NOT significant at alpha=0.05)"))
    return {"b": b, "c": c, "discordant": discordant, "p_value": p}


def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_results = {}
    per_question_hits = {}
    for name, path in VARIANTS.items():
        print(f"\n{'=' * 60}\n{name}  ({path})\n{'=' * 60}")
        db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        rag_eval.retrieve = lambda question, k, _db=db: _db.similarity_search(question, k=k)

        k5 = evaluate(TEST_SET, k=5, verbose=True)
        k3 = evaluate(TEST_SET, k=3, verbose=False)
        per_question_hits[name] = k5.pop("_per_question_hit")
        k3.pop("_per_question_hit", None)
        all_results[name] = {"k5": k5, "k3": k3}

    print(f"\n\n{'=' * 60}\nSUMMARY (k=5)\n{'=' * 60}")
    print(f"{'variant':<20}{'Recall@5':<12}{'Precision@5':<14}{'Hit@1':<10}{'MRR':<10}")
    for name, res in all_results.items():
        k5 = res["k5"]
        print(f"{name:<20}{k5['Recall@5']:<12.3f}{k5['Precision@5']:<14.3f}"
              f"{k5['Hit@1']:<10.3f}{k5['MRR']:<10.3f}")

    print(f"\n{'=' * 60}\nSIGNIFICANCE (McNemar, k=5, baseline vs each alternative)\n{'=' * 60}")
    sig = {}
    for name in ("wide_1000_200", "sentence_1000"):
        sig[name] = mcnemar(
            per_question_hits["baseline_500_50"], per_question_hits[name],
            "baseline_500_50", name,
        )

    with open("results_chunking_comparison.json", "w") as f:
        json.dump({"metrics": all_results, "significance": sig}, f, indent=2)
    print("\nWrote results_chunking_comparison.json")


if __name__ == "__main__":
    main()
