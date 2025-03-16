import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
import chromadb
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configure Streamlit page
st.set_page_config(page_title="Conversational RAG Chat", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Groq API key:", type="password")
    session_id = st.text_input("Session ID", value="session_1")
    uploaded_files = st.file_uploader("Upload PDF file(s)", type="pdf", accept_multiple_files=True)

st.title("Conversational RAG With PDF & Chat History")
st.write("Upload PDFs in the sidebar, then chat with their content.")

# Only proceed if user provided an API key
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Maintain a dictionary of ChatMessageHistory objects, keyed by session_id
    if "store" not in st.session_state:
        st.session_state.store = {}

    def get_session_history(sid: str) -> BaseChatMessageHistory:
        """Retrieve or create a ChatMessageHistory for the given session ID."""
        if sid not in st.session_state.store:
            st.session_state.store[sid] = ChatMessageHistory()
        return st.session_state.store[sid]

    # Process uploaded PDFs (if any)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split texts, create embeddings, and build retriever
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        # Prompt to reframe user question given chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question—just reformulate it if needed; otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Prompt to answer questions using retrieved context
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Wrap chain in a RunnableWithMessageHistory to store conversation
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Two-column layout: Chat (left), History (right)
        col_chat, col_history = st.columns([3, 1], gap="large")

        # ---------------
        #   Chat Column
        # ---------------
        with col_chat:
            st.header("Chat")

            # 1) Display the conversation so far
            session_history = get_session_history(session_id)
            for msg in session_history.messages:
                if msg.type == "human":
                    st.chat_message("user").write(msg.content)
                else:
                    st.chat_message("assistant").write(msg.content)

            # 2) Pinned chat input at the bottom
            user_input = st.chat_input("Your question:")
            if user_input:
                # When the user hits enter, pass input to the chain
                # This automatically appends user/assistant messages to session history
                _ = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                # Force a rerun so the conversation updates immediately
                st.rerun()

        # ---------------
        #  History Column
        # ---------------
        with col_history:
            st.header("Session History")
            with st.expander("View History", expanded=False):
                if session_history.messages:
                    for i, msg in enumerate(session_history.messages):
                        st.write(f"{i+1}. **{msg.type.upper()}**: {msg.content[:50]}...")
                else:
                    st.info("No messages yet.")

            # Clear history button
            if st.button("Clear History"):
                st.session_state.store[session_id] = ChatMessageHistory()
                st.rerun()

            # Download history if available
            if session_history.messages:
                history_text = "\n\n".join(
                    [f"{m.type.upper()}: {m.content}" for m in session_history.messages]
                )
                st.download_button(
                    label="Download History",
                    data=history_text,
                    file_name=f"chat_history_{session_id}.txt",
                    mime="text/plain"
                )

    else:
        st.info("Please upload at least one PDF to start chatting.")
else:
    st.warning("Please enter your Groq API Key in the sidebar to begin.")
