import os
import tempfile
import streamlit as st
from gtts import gTTS
from io import BytesIO
import base64
import speech_recognition as sr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

# ------------------ APP CONFIG ------------------
st.set_page_config(
    page_title="AI Medical Assistant 🩺",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <style>
        body {background-color: #f8f9fa;}
        .stChatMessage {font-size: 1.05rem;}
        .disclaimer {
            color: #c0392b;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ VECTORSTORE ------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def update_vectorstore_with_pdf(uploaded_pdf):
    """Embed new PDF dynamically"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(DB_FAISS_PATH)
    os.remove(tmp_path)
    return load_vectorstore()

# ------------------ LLM & MEMORY ------------------
def init_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=512,
        api_key=GROQ_API_KEY,
    )
    return llm

def init_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = init_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain, memory

# ------------------ TTS FUNCTION ------------------
def speak_text(text):
    tts = gTTS(text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    b64 = base64.b64encode(audio_fp.read()).decode()
    md = f"""
        <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# ------------------ VOICE INPUT ------------------
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"Recognized: {query}")
        return query
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
    except sr.RequestError:
        st.error("Speech recognition service unavailable")
    return ""

# ------------------ MAIN APP ------------------
def main():
    st.title("🩺 AI Medical Chatbot Assistant")

    # Sidebar
    st.sidebar.header("⚙️ Options")
    voice_mode = st.sidebar.checkbox("🎤 Voice Input")
    tts_mode = st.sidebar.checkbox("🔊 Speak Answers")

    st.sidebar.subheader("📄 Upload New PDF")
    uploaded_pdf = st.sidebar.file_uploader("Upload a medical PDF", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            new_db = update_vectorstore_with_pdf(uploaded_pdf)
            st.session_state.vectorstore = new_db
        st.sidebar.success("✅ PDF added successfully!")

    # Load vectorstore and chain
    vectorstore = st.session_state.get("vectorstore", load_vectorstore())
    chain, memory = init_chain(vectorstore)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Input
    if voice_mode:
        prompt = voice_to_text()
    else:
        prompt = st.chat_input("💬 Ask me anything about medical topics...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = chain.invoke({"question": prompt})
            result = response["answer"]

        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

        if tts_mode:
            speak_text(result)

        st.markdown(
            "<div class='disclaimer'>⚠️ Disclaimer: This chatbot is for educational purposes only. Always consult a licensed doctor for medical advice.</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
