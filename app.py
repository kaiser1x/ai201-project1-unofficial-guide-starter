import streamlit as st

from retriever import retrieve
from generator import generate_response

st.set_page_config(page_title="The Unofficial Guide", page_icon="🏠")

st.title("🏠 The Unofficial Guide")
st.caption(
    "RAG over residential leases, inspection checklists, and property-tax "
    "guides. Answers are grounded in the loaded documents only — if it isn't "
    "in the documents, the system says so."
)

query = st.text_input(
    "Ask a question",
    placeholder="e.g. How much can a NYC Class 1 property's assessed value rise per year?",
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving and generating..."):
        chunks = retrieve(query)
        answer = generate_response(query, chunks)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander(f"🔎 Retrieved context ({len(chunks)} chunks)"):
        if not chunks:
            st.info("No chunks retrieved.")
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**{i}. {c['source_file']}** · _{c['doc_type']}_ · "
                f"cosine distance `{c['distance']:.3f}`"
            )
            st.text(c["text"][:600])
