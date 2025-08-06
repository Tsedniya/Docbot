from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain.document_loaders import PyPDFLoader, TextLoader

import streamlit as st
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Assign only if value is not None
google_api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

if not google_api_key or not langchain_api_key:
    raise ValueError("Missing GOOGLE_API_KEY or LANGCHAIN_API_KEY in .env file.")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# LLm 
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


if not os.environ.get("WATSONX_APIKEY"):
  os.environ["WATSONX_APIKEY"] = getpass.getpass("Enter API key for IBM watsonx: ")

embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
)

# streamlit framework
st.title('Ask Bot')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    print(message)
    st.chat_message(message['role']).markdown(message['content'])

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file and 'vectorstore' not in st.session_state:
    # Load document
    if uploaded_file.type == "application/pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
    else:
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.read())
        loader = TextLoader("temp.txt")

    documents = loader.load()

    # Split and embed
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = vectorstore
    st.success("Document uploaded and indexed!")

prompt = st.chat_input('pass your prompt here')

if prompt:
    user_input = prompt
    st.chat_message('user').markdown(user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    chat_history = st.session_state.messages[-5:] if 'messages' in st.session_state else []
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the retrieved context from the document to answer. Respond conversationally."),
        ("system", "Chat history:\n" + history_str),
        ("user", "Question: {question}")
    ])

    with st.spinner("Thinking..."):
        if 'vectorstore' in st.session_state:
            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", k=3)
            relevant_docs = retriever.get_relevant_documents(user_input)
            context_text = "\n".join([doc.page_content for doc in relevant_docs])
            full_prompt = prompt_template.invoke({"question": f"{user_input}\n\nContext:\n{context_text}"})
            response = model.invoke(full_prompt)
        else:
            # No vectorstore, fallback to chat only
            fallback_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Respond conversationally."),
                ("system", "Chat history:\n" + history_str),
                ("user", "Question: {question}")
            ])
            full_prompt = fallback_template.invoke({"question": user_input})
            response = model.invoke(full_prompt)

    st.chat_message('assistant').markdown(response.content)
    st.session_state.messages.append({'role': 'assistant', 'content': response.content})