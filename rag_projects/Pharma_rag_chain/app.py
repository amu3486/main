# app.py
import streamlit as st
import os
from dotenv import load_dotenv  # For loading API key from .env file (optional but good practice)

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
CHROMADB_PERSIST_DIR = "chroma_db"
CHROMADB_COLLECTION_NAME = "pharma_docs_collection"


# --- Helper Functions (from previous steps, adapted for Streamlit) ---

@st.cache_resource  # Cache the embedding model to avoid reloading on every rerun
def get_embedding_model(google_api_key: str):
    if not google_api_key:
        st.error("Google API Key is not set. Please enter it in the sidebar.")
        return None
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error initializing embedding model: {e}")
        return None


@st.cache_resource  # Cache the LLM for chat
def get_chat_model(google_api_key: str):
    if not google_api_key:
        st.error("Google API Key is not set. Please enter it in the sidebar.")
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.2)
    except Exception as e:
        st.error(f"Error initializing chat model: {e}")
        return None


def load_and_split_documents(pdf_file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Loads a single PDF document, splits it into chunks, and returns LangChain Document objects.
    """
    print(f"Loading {pdf_file_path}...")
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(split_documents)} chunks.")
    return split_documents


def create_or_update_chroma_collection(
        documents: list[Document],
        embedding_model,
        collection_name: str = CHROMADB_COLLECTION_NAME,
        persist_directory: str = CHROMADB_PERSIST_DIR
):
    """
    Creates or loads a ChromaDB collection and optionally adds new documents.
    Returns the updated vectorstore.
    """
    if not embedding_model:
        st.error("Embedding model not initialized. Cannot update database.")
        return None

    # Initialize ChromaDB client. If persist_directory exists, it will load the existing DB.
    # Pass collection_metadata if you want to ensure the embedding function is loaded correctly.
    # Chroma handles adding the embedding function during initialization when reading from disk.
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    if documents:
        print(f"Adding {len(documents)} new chunks to ChromaDB collection '{collection_name}'...")
        # Chroma's `add_documents` method automatically handles embedding generation
        vectorstore.add_documents(documents)
        print("Documents added successfully.")
        # Persist the changes to disk
        vectorstore.persist()
        print(f"ChromaDB persisted to {persist_directory}")
        st.success(f"Added {len(documents)} document chunks to your knowledge base!")
    else:
        print(f"No new documents provided. Loaded existing ChromaDB collection '{collection_name}'.")

    return vectorstore


def get_rag_chain(vectorstore, chat_model):
    """Initializes and returns the LangChain ConversationalRetrievalChain."""
    if not vectorstore or not chat_model:
        return None

    # Use a custom prompt to guide the LLM
    custom_template = """You are a helpful pharmaceutical research assistant. 
    Answer the user's question based ONLY on the provided context. 
    If you don't know the answer, state that you don't know. 
    Keep your answers concise and accurate.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    Answer:"""
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key = 'answer')

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
        return_source_documents=True, # To show which documents were used
        output_key="answer"
    )
    return rag_chain


# --- Streamlit UI ---
st.set_page_config(page_title="PharmaQuery AI", layout="wide")

st.title("🔬 PharmaQuery AI")
st.markdown("Your intelligent assistant for pharmaceutical research insights.")

# --- Sidebar for API Key and Document Upload ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input
    # Try to load from env var first, then session state
    initial_api_key = os.getenv("GOOGLE_API_KEY", st.session_state.get("google_api_key", ""))
    google_api_key = st.text_input(
        "Enter your Google Gemini API Key:",
        type="password",
        value=initial_api_key,
        key="google_api_key_input"  # Unique key for this widget
    )
    # Store in session state for persistence
    st.session_state["google_api_key"] = google_api_key

    st.markdown("---")
    st.header("Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here or click to browse",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents and Update Database"):
        if not st.session_state.get("google_api_key"):
            st.error("Please enter your Google API Key before processing documents.")
        elif uploaded_files:
            with st.spinner("Processing documents and updating knowledge base..."):
                embedding_model = get_embedding_model(st.session_state.google_api_key)
                if embedding_model:
                    all_split_documents = []
                    # Create a temporary directory to save uploaded PDFs
                    os.makedirs("temp_uploads", exist_ok=True)
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("temp_uploads", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        split_docs = load_and_split_documents(file_path)
                        all_split_documents.extend(split_docs)
                        os.remove(file_path)  # Clean up temp file

                    if all_split_documents:
                        st.session_state.vectorstore = create_or_update_chroma_collection(
                            all_split_documents, embedding_model
                        )
                    else:
                        st.warning("No valid text extracted from uploaded PDFs.")
                else:
                    st.error("Failed to initialize embedding model.")
        else:
            st.warning("Please upload at least one PDF file.")

# --- Initialize or Load ChromaDB and RAG Chain ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # For chat history

# Attempt to load existing ChromaDB or initialize if API key is present
if st.session_state.google_api_key and st.session_state.vectorstore is None:
    with st.spinner("Loading existing knowledge base..."):
        embedding_model = get_embedding_model(st.session_state.google_api_key)
        if embedding_model:
            # Load without adding new documents
            st.session_state.vectorstore = create_or_update_chroma_collection([], embedding_model)
        else:
            st.warning("Could not load knowledge base without valid API key.")

# Initialize RAG chain once vectorstore and chat model are ready
if st.session_state.vectorstore and st.session_state.rag_chain is None:
    with st.spinner("Initializing AI..."):
        chat_model = get_chat_model(st.session_state.google_api_key)
        if chat_model:
            st.session_state.rag_chain = get_rag_chain(st.session_state.vectorstore, chat_model)
        else:
            st.error("Failed to initialize chat model. Please check your API key.")

# --- Main Chat Interface ---
st.subheader("Ask PharmaQuery:")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for new query
if query := st.chat_input("Enter your query about pharmaceutical research..."):
    if not st.session_state.google_api_key:
        st.error("Please enter your Google API Key in the sidebar to start chatting.")
    elif not st.session_state.rag_chain:
        st.error(
            "AI is not initialized. Please ensure your API key is correct and try processing documents or reloading the page.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Fetching insights..."):
                try:
                    # Invoke the RAG chain
                    response = st.session_state.rag_chain.invoke({"question": query})

                    # Extract answer and source documents
                    answer = response["answer"]
                    source_documents = response.get("source_documents", [])

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    if source_documents:
                        with st.expander("Source Documents"):
                            for i, doc in enumerate(source_documents):
                                st.write(f"**Document {i + 1}:**")
                                st.write(f"Source: `{doc.metadata.get('source', 'N/A')}`")
                                st.write(f"Page: `{doc.metadata.get('page', 'N/A')}`")
                                with st.expander("Show content"):
                                    st.text(doc.page_content)
                                st.markdown("---")

                except Exception as e:
                    st.error(f"An error occurred while processing your query: {e}")
                    st.warning("Please check your API key and ensure the knowledge base is loaded correctly.")