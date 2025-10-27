import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Offline RAG Bot", layout="wide")

st.title("🤖 RAG Application (No OpenAI)")
st.markdown("Chat with your documents using **local embeddings + Hugging Face model**")

# --- Directories ---
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("chroma_db"):
    os.makedirs("chroma_db")

# --- File Upload ---
uploaded_files = st.file_uploader("📂 Upload PDF or TXT files", accept_multiple_files=True)

# --- Load or Process Documents ---
if uploaded_files:
    st.info("Processing uploaded files...")

    documents = []
    for file in uploaded_files:
        file_path = os.path.join("uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        docs = loader.load()
        documents.extend(docs)

    # --- Split into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(documents)

    # --- Create Local Embeddings ---
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- Store in ChromaDB ---
    vectorstore = Chroma.from_documents(docs_chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    st.success("✅ Files processed and stored in ChromaDB!")

else:
    # Load existing database if available
    if os.path.exists("chroma_db") and os.listdir("chroma_db"):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    else:
        vectorstore = None

# --- Setup Local LLM (Hugging Face Model) ---
if vectorstore:
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Free text generation model
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)
else:
    chain = None
    st.warning("Upload a file first to initialize RAG pipeline.")

# --- Chat Section ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi Amit 👋! Upload a file and ask me anything about it."}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"<div style='background-color:#E8EAF6; padding:10px; border-radius:10px;'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; text-align:right;'>🧑 {msg['content']}</div>", unsafe_allow_html=True)

user_input = st.text_input("💬 Ask your question:")

if user_input and chain:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = chain({"question": user_input})
        answer = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(f"<div style='background-color:#E8EAF6; padding:10px; border-radius:10px;'>🤖 {answer}</div>", unsafe_allow_html=True)

# --- Hide Streamlit Menu ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)
