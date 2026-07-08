# This is the main application file that runs the Flask web server and handles user requests
# Which will include uploading files, chunking them, embedding the chunks, storing them in a vector database, and searching for similar chunks based on user queries.
'''
Interview question: "Why st.session_state?"
Answer: Streamlit reruns the script on every interaction. Without session state,
we'd re-index the vector DB on every keystroke. Session state persists objects across reruns.
'''

import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(page_title="My Notes RAG", page_icon="📝")

st.title("Chat with Your Notes")
st.markdown("A local RAG chatbot powered by Ollama + ChromaDB")

# Initialize pipeline once
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
    with st.spinner("Indexing your notes..."):
        st.session_state.pipeline.index()

question = st.text_input("Ask something about your notes:")
if question:
    with st.spinner("Thinking..."):
        result = st.session_state.pipeline.ask(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("See source chunks"):
        for source in result["sources"]:
            st.markdown(f"**{source['metadata']['source']}** (distance: {source['distance']:.3f})")
            st.text(source["text"][:300] + "...")
