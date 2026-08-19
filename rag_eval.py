"""
Retrieval evaluation harness for the Gale Encyclopedia RAG system.

GROUND TRUTH: PAGE IDENTITY, NOT CHUNK CONTENTS
------------------------------------------------
Earlier version of this harness judged a retrieved chunk relevant if it
*contained* a hand-picked answer phrase. That's a confounded ground truth
for a chunking comparison: change the chunk boundaries and the same source
text gets sliced differently, so whether a phrase survives intact inside
one particular chunk depends on the very thing being varied.

This version judges relevance by the PDF page a chunk came from
(`doc.metadata["page"]`, set by PyPDFLoader, 0-indexed). Page identity is
fixed by the source PDF and does not change when chunk_size or overlap
change — so the ground truth stays constant while only the retrieval
system varies. answer_keys are kept on each TestCase for human-readable
documentation of what the question is actually about, but they are not
used for scoring.

TEST SET
--------
51 questions, each hand-written from a real page of the actual indexed PDF
(data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf), spread across pages
20-700 for topic variety, source_page recorded for each.

Run: python rag_eval.py
"""

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ----------------------------------------------------------------------
# 1. TEST SET - 51 real questions from the indexed PDF
# ----------------------------------------------------------------------

@dataclass
class TestCase:
    question: str
    source_page: int  # 0-indexed PDF page the question was written from
    answer_keys: list = field(default_factory=list)  # documentation only


TEST_SET = [
    TestCase("What is a gene, in terms of DNA?", 20, ["small part of DNA"]),
    TestCase("What is a tumor also called in medical terminology?", 20, ["neoplasm"]),
    TestCase("What percentage of cataract surgeries have no complications?", 60,
             ["90%", "no complications"]),
    TestCase("What virus causes approximately 90% of all cervical cancers?", 100,
             ["human papilloma virus", "90%"]),
    TestCase("How effective is the Varivax vaccine at preventing severe cases of chickenpox?", 140,
             ["Varivax", "100%"]),
    TestCase("What is total serum cholesterol commonly abbreviated as?", 180,
             ["Total serum cholesterol", "TC"]),
    TestCase("What neurotransmitter's re-uptake does cocaine block?", 220,
             ["dopamine", "re-uptake"]),
    TestCase("What brain region does cocaine affect that connects to the nucleus accumbens?", 220,
             ["ventral tegmental area", "nucleus accumbens"]),
    TestCase("What should be taken alongside zinc lozenges to help with a cold, per the encyclopedia's nutritional therapy suggestion?", 260,
             ["zinc lozenges", "vitamin C"]),
    TestCase("How often is Depo Provera injected as a birth control method?", 300,
             ["Depo Provera", "every three months"]),
    TestCase("What disease is variant Creutzfeldt-Jakob disease (vCJD) believed to be caused by?", 340,
             ["vCJD", "BSE"]),
    TestCase("What is the male to female ratio for UTIs in adult life?", 380, ["1:50"]),
    TestCase("How many people does depressive disorder affect each year in the United States, per the encyclopedia?", 420,
             ["17 million"]),
    TestCase("What plant are digitalis drugs derived from?", 460, ["foxglove"]),
    TestCase("How accurate is FISH for diagnosing DiGeorge syndrome before birth?", 460,
             ["95%", "FISH"]),
    TestCase("What procedure involves inserting a camera inside a thin tube directly into the uterus?", 500,
             ["hysterscopy"]),
    TestCase("What condition results from excessively high sodium levels?", 540, ["hypernatremia"]),
    TestCase("What percentage of hospitalized patients develop hyponatremia?", 540,
             ["1%", "hyponatremia"]),
    TestCase("Name two NSAIDs used to relieve pain from endometriosis.", 580,
             ["ibuprofen", "naproxen"]),
    TestCase("Where is erythropoietin primarily produced in the body?", 620, ["kidneys"]),
    TestCase("What is another medical term for face lift surgery?", 660, ["rhytidoplasty"]),
    TestCase("What are fever-inducing agents called?", 700, ["pyrogens"]),
    TestCase("What does the World Health Organization's analgesic ladder recommend starting with for mild pain?", 30,
             ["analgesic ladder", "non-opioid"]),
    TestCase("What skin color change is an unreliable symptom of carbon monoxide poisoning?", 40,
             ["cherry red", "bright pink"]),
    TestCase("Why should CPR never be performed on a healthy person?", 50,
             ["serious injury", "beating heart"]),
    TestCase("What deficiency can malabsorption from celiac disease cause anemia from?", 70,
             ["iron deficiency"]),
    TestCase("What is agranulocytosis, according to the encyclopedia's key terms?", 80,
             ["granulocyte-producing bone marrow"]),
    TestCase("What has changed in the causes of cerebral palsy over the last 20-30 years, according to the encyclopedia?", 90,
             ["Rh disease", "prematurity"]),
    TestCase("What class of medication, originally developed to treat depression, is also prescribed for chronic neck pain?", 110,
             ["tricyclic antidepressants"]),
    TestCase("What should not be given together with activated charcoal because it will adsorb it?", 120,
             ["syrup of ipecac"]),
    TestCase("Name two anti-emetic medications used to lessen nausea from chemotherapy.", 130,
             ["Kytril", "Zofran"]),
    TestCase("How many doses of the MMR vaccine are recommended in childhood?", 150,
             ["Measles, Mumps, Rubella", "One dose"]),
    TestCase("How is Chlamydia psittaci prevented, according to the encyclopedia?", 160, ["birds"]),
    TestCase("What does the World Health Organization's oral rehydration fluid for cholera contain?", 170,
             ["water, sugar, and salts"]),
    TestCase("What is another name for chronic fatigue syndrome mentioned in the encyclopedia?", 190,
             ["myalgic encephalomyelitis"]),
    TestCase("What is the median survival for COPD patients after diagnosis?", 200, ["10 years"]),
    TestCase("What is clubfoot?", 210, ["twisted into an abnormal position"]),
    TestCase("Who developed Rational Emotive Behavior Therapy (REBT) and in what year?", 230,
             ["Albert Ellis", "1955"]),
    TestCase("What defines Stage I colon cancer?", 240, ["confined to the epithelium"]),
    TestCase("What is the purpose of a loop colostomy?", 250, ["temporary stoma"]),
    TestCase("What is the effectiveness rate of male condoms for preventing pregnancy when used correctly?", 270,
             ["90%"]),
    TestCase("What vitamin supplement has been shown to reduce the incidence of neural tube defects?", 280,
             ["Folic acid"]),
    TestCase("What surgical procedure corrects transposition of the great arteries by connecting the aorta to the left ventricle?", 290,
             ["Arterial switch"]),
    TestCase("What bacterium can contaminate eyedrops and cause corneal infection?", 310,
             ["Pseudomonas aeruginosa"]),
    TestCase("What percentage of patients experience full relief from angina after coronary artery bypass surgery?", 320,
             ["70%"]),
    TestCase("What class of blood pressure medication can cause coughing as a side effect?", 330,
             ["ACE inhibitors"]),
    TestCase("What is Type I cryoglobulin also called?", 350, ["monoclonal cryoglobulinemia"]),
    TestCase("What hormone is secreted in excess in Cushing's syndrome?", 360, ["Cortisol"]),
    TestCase("What medication class can worsen a patient's condition if used to treat dysentery?", 370,
             ["Anti-motility medications", "loperamide"]),
    TestCase("What is dacryocystitis?", 390, ["inflammation of the tear sac"]),
    TestCase("What decongestant ingredient did the FDA prohibit from over-the-counter sales in 2000?", 400,
             ["phenylpropanolamine"]),
]


# ----------------------------------------------------------------------
# 2. RETRIEVER - the actual FAISS index built by memory.py
# ----------------------------------------------------------------------

DB_FAISS_PATH = "vectorstore/db_faiss"

_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_db = FAISS.load_local(DB_FAISS_PATH, _embedding_model, allow_dangerous_deserialization=True)


def retrieve(question: str, k: int) -> list:
    """Return the top-k retrieved LangChain Documents (content + metadata,
    including the source page)."""
    return _db.similarity_search(question, k=k)


# ----------------------------------------------------------------------
# 3. SCORING - relevance is page identity, not chunk contents
# ----------------------------------------------------------------------

def is_relevant(doc, case: TestCase) -> bool:
    return doc.metadata.get("page") == case.source_page


def evaluate(test_set, k=5, verbose=True):
    n = len(test_set)
    if n == 0:
        raise ValueError("TEST_SET is empty - add your questions first.")

    hits_at_k = 0
    hits_at_1 = 0
    precision_sum = 0.0
    reciprocal_ranks = 0.0
    failures = []
    per_question_hit = {}  # question -> bool, for paired significance tests

    for case in test_set:
        docs = retrieve(case.question, k)
        flags = [is_relevant(d, case) for d in docs]

        hit = any(flags)
        per_question_hit[case.question] = hit

        if hit:
            hits_at_k += 1
            rank = flags.index(True) + 1
            reciprocal_ranks += 1.0 / rank
            if rank == 1:
                hits_at_1 += 1
        else:
            failures.append(case.question)

        precision_sum += sum(flags) / len(docs) if docs else 0.0

    results = {
        f"Recall@{k}": hits_at_k / n,
        f"Precision@{k}": precision_sum / n,
        "Hit@1": hits_at_1 / n,
        "MRR": reciprocal_ranks / n,
        "n_questions": n,
        "_per_question_hit": per_question_hit,
    }

    if verbose:
        print(f"\nEvaluated {n} questions at k={k}\n" + "-" * 34)
        for key, val in results.items():
            if key in ("n_questions", "_per_question_hit"):
                continue
            print(f"{key:<14} {val:.3f}")
        if failures:
            print(f"\n{len(failures)} misses:")
            for q in failures:
                print(f"  - {q}")

    return results


if __name__ == "__main__":
    import json

    results_k5 = evaluate(TEST_SET, k=5)
    print()
    results_k3 = evaluate(TEST_SET, k=3)

    def strip(r):
        return {k: v for k, v in r.items() if k != "_per_question_hit"}

    with open("results_rag_eval.json", "w") as f:
        json.dump({"k5": strip(results_k5), "k3": strip(results_k3)}, f, indent=2)
    print("\nWrote results_rag_eval.json")
