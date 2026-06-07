from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

# Grounding is enforced primarily here. The model is told to answer ONLY from
# the supplied context, to cite the source_file behind each claim, and to
# refuse when the context does not contain the answer.
SYSTEM_PROMPT = """You are The Unofficial Guide, a retrieval-augmented assistant for real estate, property management, and property-tax questions.

STRICT GROUNDING RULES:
1. Answer using ONLY the information in the CONTEXT below. Do not use any outside or prior knowledge.
2. If the answer is not present in the context, reply exactly: "I don't have that information in the loaded documents." Do not guess.
3. Cite the source for every factual claim, using the source filename shown in the context, e.g. (Source: Property-Tax-Guide-NYC-Class-One.pdf).
4. If the context is unrelated to the question, treat it as no answer and refuse per rule 2.
5. Be concise and quote specific numbers, thresholds, or phrasing from the context when relevant."""


def _format_context(chunks):
    """Render retrieved chunks into a numbered, source-labeled context block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[{i}] Source: {c['source_file']} | Type: {c['doc_type']}\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved chunks.

    Returns a plain string. Refuses (no LLM call) when retrieval is empty;
    otherwise the system prompt enforces grounding and source attribution.
    """
    if not retrieved_chunks:
        return "I don't have that information in the loaded documents."

    context = _format_context(retrieved_chunks)
    user_message = f"CONTEXT:\n{context}\n\nQUESTION: {query}"

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # low — grounded extraction, not creative writing
    )
    return response.choices[0].message.content
