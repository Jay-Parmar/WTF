import os
import time
import zipfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import traceback
import tempfile

from utils.pdf_table_extract import extract_tables_from_pdf
# from utils.pdf_ocr import extract_text_from_any_pdf  # Uncomment if using OCR fallback

# --- Setup
load_dotenv()
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
st.set_page_config(page_title="Tender Extraction & PDF Table App", layout="wide")

Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "built" not in st.session_state:
    st.session_state.built = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "timings" not in st.session_state:
    st.session_state.timings = {}

ALLOWED_EXTS = [".pdf", ".docx", ".txt", ".xlsx", ".xls", ".csv"]

def clean_data_folder():
    for f in DATA_DIR.glob("*"):
        if f.suffix.lower() not in ALLOWED_EXTS:
            f.unlink()

@st.cache_data(show_spinner=False)
def load_docs(dir_path: Path):
    files = [f for f in dir_path.glob("*") if f.suffix.lower() in ALLOWED_EXTS]
    splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=600)
    # --- If you want OCR fallback for PDFs, load each PDF with extract_text_from_any_pdf, build Document objects.
    # return [Document(extract_text_from_any_pdf(str(f)), metadata={"file_path": str(f)}) if f.suffix.lower() == ".pdf" else ... for f in files], splitter
    return SimpleDirectoryReader(input_files=files).load_data(), splitter

@st.cache_resource(show_spinner=False)
def build_index(_docs, _splitter, _storage_dir: Path):
    idx = VectorStoreIndex.from_documents(_docs, node_parser=_splitter)
    idx.storage_context.persist(persist_dir=str(_storage_dir))
    return idx

@st.cache_resource(show_spinner=False)
def load_index(storage_dir: Path):
    ctx = StorageContext.from_defaults(persist_dir=str(storage_dir))
    return load_index_from_storage(ctx)

# --- TABS for two main functionalities ---
tab1, tab2 = st.tabs(["📑 Tender Extraction Q&A (RAG)", "🗃️ PDF Table Extractor"])

with tab1:
    st.title("📑 Tender Extraction Q&A – Modern RAG (Groq + LlamaIndex)")
    st.sidebar.header("Upload & Settings")
    uploaded_files = st.sidebar.file_uploader(
        "Upload ZIP / PDF / XLSX / CSV / DOCX / TXT",
        type=["zip", "pdf", "xlsx", "xls", "csv", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            dest = DATA_DIR / uploaded.name
            if dest.exists():
                st.sidebar.warning(f"{uploaded.name} already exists; skipping.")
            else:
                try:
                    dest.write_bytes(uploaded.getbuffer())
                    st.sidebar.success(f"Uploaded {uploaded.name}")

                    if uploaded.name.lower().endswith(".zip"):
                        with zipfile.ZipFile(dest, "r") as zip_ref:
                            zip_ref.extractall(DATA_DIR)
                        dest.unlink()
                        st.sidebar.success(f"Extracted and removed {uploaded.name}")
                except Exception as e:
                    st.sidebar.error(f"Failed to upload {uploaded.name}: {e}")
        clean_data_folder()

    if st.sidebar.button("🔨 Build Index"):
        with st.spinner("Building index…"):
            try:
                clean_data_folder()
                docs, splitter = load_docs(DATA_DIR)
                if not docs:
                    st.sidebar.error("No documents found; upload first.")
                    st.session_state.built = False
                else:
                    st.write(f"Loaded {len(docs)} documents.")
                    t0 = time.perf_counter()
                    idx = build_index(docs, splitter, STORAGE_DIR)
                    st.session_state.timings["index_build"] = time.perf_counter() - t0
                    st.session_state.built = True
                    st.sidebar.success("✅ Index built!")
            except Exception as e:
                st.sidebar.error(f"Failed to build index: {e}")

    if st.sidebar.button("🗑️ Clear All"):
        try:
            for d in (DATA_DIR, STORAGE_DIR):
                for f in d.glob("*"):
                    f.unlink()
            st.session_state.built = False
            st.session_state.messages = []
            st.sidebar.success("Cleared everything.")
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            elif hasattr(st, "rerun"):
                st.rerun()
            else:
                st.sidebar.info("Please refresh the page to apply changes.")
        except Exception as e:
            st.sidebar.error(f"Failed to clear files: {e}")

    index = None
    if st.session_state.built:
        try:
            index = load_index(STORAGE_DIR)
            st.info("Index loaded from storage.")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

    st.header("Ask about your tender files")
    st.markdown("> **Tip:** For best results, try asking: `Copy the full table under Clause 3.5 Pre-Qualification Criteria from the document.`")

    with st.form("query_form"):
        query = st.text_input("Your question:")
        submit = st.form_submit_button("Ask")

    if submit:
        if not index:
            st.error("Please build the index first.")
        elif not query.strip():
            st.warning("Type a question before hitting Ask.")
        else:
            with st.spinner("Retrieving answer…"):
                try:
                    engine = index.as_query_engine()
                    t0 = time.perf_counter()
                    resp = engine.query(query)
                    st.session_state.timings["query_time"] = time.perf_counter() - t0
                    st.session_state.messages.append({
                        "query": query,
                        "response": resp.response,
                        "sources": resp.source_nodes,
                    })
                except Exception as e:
                    st.error(f"Error during retrieval: {e}")
                    st.text(traceback.format_exc())

    if st.session_state.messages:
        st.header("Conversation History")
        for msg in st.session_state.messages:
            st.markdown(f"**Q: {msg['query']}**")
            st.success(msg["response"])
            with st.expander("Source Chunks (raw text, for copy-paste):"):
                for i, n in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    chunk = n.node.get_content()
                    st.code(chunk)
                    st.button(
                        "Copy to clipboard",
                        key=f"copy_btn_{i}_{msg['query']}",
                        help="Copy the full source chunk for reference.",
                        disabled=True
                    )

    with st.expander("📂 Uploaded Files"):
        files = [f for f in DATA_DIR.glob("*") if f.suffix.lower() in ALLOWED_EXTS]
        if files:
            for f in files:
                st.write(f"• {f.name} ({f.stat().st_size // 1024} KB)")
        else:
            st.write("No uploads yet.")

    with st.expander("⚙️ Debug Info"):
        st.write("Index built:", st.session_state.built)
        st.write("Messages count:", len(st.session_state.messages))
        st.write("Timings (s):", st.session_state.timings)

with tab2:
    st.title("🗃️ PDF Table Extractor")
    st.markdown("""
    Upload a PDF file containing tables.
    Tables will be extracted and shown below.
    Powered by Camelot (with pdfplumber fallback).
    """)
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_table_upload")
    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getbuffer())
            tmp_path = tmp.name
        st.info("Extracting tables, please wait...")
        tables = extract_tables_from_pdf(tmp_path)
        if not tables:
            st.warning("No tables found in the PDF.")
        else:
            st.success(f"Found {len(tables)} table(s)!")
            for i, table_df in enumerate(tables):
                st.subheader(f"Table {i+1}")
                st.dataframe(table_df)
                csv = table_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download this table as CSV",
                    data=csv,
                    file_name=f"table_{i+1}.csv",
                    mime="text/csv"
                )
    else:
        st.info("Please upload a PDF file to extract tables.")
