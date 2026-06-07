"""
Evaluation + demo harness. Produces the real outputs needed for the README:
  - retrieval results (query -> top-k chunks with source + distance)
  - generated answers with source attribution
  - one out-of-scope query showing the refusal

Run AFTER `python build_index.py`:

    python run_eval.py
"""
from retriever import retrieve
from generator import generate_response

# 5 evaluation questions from planning.md.
EVAL_QUESTIONS = [
    "What is the maximum Social Security benefit amount used for retirement income exclusions under Georgia property tax senior exemptions in 2014?",
    "In New York City, how much can the assessed value of a Class 1 property increase from one year to the next according to state law caps?",
    "What is the late fee penalty in the Basic Rental Agreement, and after what day is rent considered late?",
    "According to the HUD Housing Choice Voucher checklist, what size threshold determines if deteriorated interior paint fails inspection?",
    "What method does the Smartsheet Property Management checklist recommend for checking windows for hidden air leaks?",
]

# Deliberately outside the corpus -> should trigger the refusal.
OUT_OF_SCOPE = "How do I overclock my laptop's GPU for gaming?"


def show(query):
    chunks = retrieve(query)
    print("=" * 80)
    print(f"QUERY: {query}")
    print("-" * 80)
    print(f"RETRIEVED {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] dist={c['distance']:.3f}  {c['source_file']}  ({c['doc_type']})")
        print(f"      {c['text'][:160].strip()!r}")
    print("-" * 80)
    print("ANSWER:")
    print(generate_response(query, chunks))
    print()


if __name__ == "__main__":
    print("\n########## EVALUATION QUESTIONS ##########\n")
    for q in EVAL_QUESTIONS:
        show(q)

    print("\n########## OUT-OF-SCOPE (refusal) ##########\n")
    show(OUT_OF_SCOPE)
