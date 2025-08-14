import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import re
from typing import List, Dict, Tuple, Optional
import requests
from sentence_transformers import SentenceTransformer
from streamlit_lottie import st_lottie


# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="StudyMate - AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------- Load Lottie Animation ---------------------
def load_lottieurl(url: str):
    """Load a Lottie animation from a URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


animation_url = "https://assets5.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_animation = load_lottieurl(animation_url)


# --------------------- Model Caching ---------------------
@st.cache_resource
def get_embedding_model():
    """Load the sentence transformer embedding model once."""
    return SentenceTransformer('all-MiniLM-L6-v2')


# --------------------- Core App Class ---------------------
class StudyMate:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []

    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """Extract text chunks from a PDF file."""
        data = pdf_file.getvalue()
        doc = fitz.open(stream=data, filetype="pdf")
        text_chunks: List[Dict] = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
                for para in paragraphs:
                    if len(para) > 100:
                        text_chunks.append({
                            "text": para,
                            "source": getattr(pdf_file, "name", "uploaded.pdf"),
                            "page": page_num + 1
                        })
        doc.close()
        return text_chunks

    def create_embeddings_and_index(self, text_chunks: List[Dict]) -> int:
        """Create FAISS index from extracted text chunks."""
        if not text_chunks:
            return 0

        texts = [c["text"] for c in text_chunks]

        with st.spinner("🔍 Creating embeddings..."):
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False
            )

        embeddings = embeddings.astype("float32", copy=False)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.chunks = texts
        self.chunk_metadata = text_chunks
        return len(texts)

    def search_similar_chunks(self, query: str, k: int = 3) -> List[Tuple[str, Dict]]:
        """Search for chunks most similar to the query."""
        if self.embedding_model is None or self.index is None or not self.chunks:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32", copy=False)

        k = max(1, min(k, len(self.chunks)))
        distances, indices = self.index.search(query_embedding, k)

        results: List[Tuple[str, Dict]] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], self.chunk_metadata[idx]))
        return results

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate a basic answer from retrieved chunks."""
        context = "\n\n".join(context_chunks[:3])
        if not context.strip():
            return "❌ I couldn't find relevant information in your documents to answer that question."
        return (
            "### ✅ Based on your study materials:\n\n"
            f"*Answer*: {context[:500]}...\n\n"
            "Note: This is a basic response. In production, this will be enhanced with IBM Watson or similar LLMs."
        )


# --------------------- UI / Main Function ---------------------
def main():
    st.title("📚 StudyMate")
    st.subheader("Your AI-Powered PDF Study Assistant")

    # Header Animation
    if lottie_animation:
        st_lottie(lottie_animation, height=200, key="header_animation")

    st.markdown("Upload your study materials and ask questions in natural language!")

    if "studymate" not in st.session_state:
        st.session_state.studymate = StudyMate()

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    with st.sidebar:
        st.header("📄 Upload Study Materials")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload textbooks, lecture notes, or research papers"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            if st.button("📘 Process Documents"):
                with st.spinner("Extracting text from PDFs..."):
                    all_chunks = []
                    for file in uploaded_files:
                        try:
                            chunks = st.session_state.studymate.extract_text_from_pdf(file)
                            all_chunks.extend(chunks)
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")

                    if all_chunks:
                        count = st.session_state.studymate.create_embeddings_and_index(all_chunks)
                        st.success(f"🎉 Successfully processed {count} text chunks!")
                        st.session_state.documents_processed = True
                    else:
                        st.error("❌ No readable text found in uploaded documents.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")

        if st.session_state.get("documents_processed"):
            user_question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What are the key findings of this paper?",
                key="question_input"
            )

            if st.button("🔎 Ask StudyMate") and user_question:
                with st.spinner("Searching your materials..."):
                    results = st.session_state.studymate.search_similar_chunks(user_question, k=5)
                    if results:
                        context_texts = [text for text, _ in results]
                        answer = st.session_state.studymate.generate_answer(user_question, context_texts)
                        st.markdown(answer)
                        with st.expander("📖 Sources"):
                            for i, (text, meta) in enumerate(results[:3]):
                                st.markdown(f"*Source {i+1}*: {meta['source']} (Page {meta['page']})")
                                snippet = text if len(text) < 300 else text[:300] + "..."
                                st.text(snippet)
                                st.markdown("---")
                    else:
                        st.warning("😕 No relevant results found. Try rephrasing your question.")
        else:
            st.info("👈 Upload and process documents before asking questions.")

    with col2:
        st.header("📊 Study Session")
        if st.session_state.get("documents_processed"):
            st.metric("Text Chunks", len(st.session_state.studymate.chunks))

        st.markdown("### 📝 Study Tips")
        st.markdown(
            "- Ask specific, focused questions\n"
            "- Try different phrasings\n"
            "- Use keywords from the material\n"
            "- Ask for examples or summaries"
        )

        with st.expander("🔧 Tech Stack"):
            st.markdown(
                "*Current Demo:*\n"
                "- PyMuPDF (PDF text extraction)\n"
                "- SentenceTransformers (embeddings)\n"
                "- FAISS (semantic search)\n"
                "- Streamlit (interface)\n\n"
                "*Future Enhancements:*\n"
                "- IBM Watson LLM / Mixtral-8x7B\n"
                "- More robust chunking\n"
                "- OCR & scanned PDF support"
            )


# --------------------- Entry Point ---------------------
if __name__ == "__main__":
    main()
