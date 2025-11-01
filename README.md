🩺 AI Medical Chatbot Assistant

AI Medical Chatbot Assistant — an advanced AI-powered healthcare chatbot built using Groq LLM, LangChain, and Streamlit. This project uses Retrieval-Augmented Generation (RAG) for accurate medical responses, integrates Hugging Face embeddings and FAISS vector search, and supports voice input and text-to-speech via gTTS or ElevenLabs. Designed to assist users with medical information in a natural, conversational way, it demonstrates real-world applications of Generative AI, Natural Language Processing (NLP), and Machine Learning in digital healthcare.





*Features

 -Chat-based medical question answering

- Optional voice input (speech-to-text)

- Text-to-speech output using gTTS

 -Context-aware RAG responses (Groq + FAISS)

 -Streamlit interface with a clean and modern design



*Technologies Used

Python 3.10+
Streamlit — frontend UI
LangChain — RAG and LLM integration
Groq API (Llama 3.1 8B) — backend language model
HuggingFace Sentence Transformers — embeddings
FAISS — vector storage and retrieval
gTTS — text-to-speech conversion
SpeechRecognition — voice input
uv — environment and dependency manager


-Installation & Setup (using uv)

1)Clone this repository
code:
git clone https://github.com/yourusername/ai-medical-chatbot.git
cd ai-medical-chatbot

2)Install dependencies using uv
code:
uv sync
(This installs all packages listed in your uv.lock)

3)Set up environment variables

-Create a .env file in the project root:
(in the .env file store your groq llm api)
GROQ_API_KEY=your_groq_api_key_here

4)Run the Streamlit app

(write in terminal this):
uv run streamlit run chatbot.py (or your file name)



*How It Works

1)Medical documents are pre-embedded using HuggingFace embeddings.

2)These embeddings are stored in a FAISS database.

3)When a user asks a question, the Groq LLM retrieves relevant chunks and generates context-based answers.

4)Optionally, the chatbot can speak responses aloud using gTTS.


⚠️ Disclaimer

This chatbot is built for educational and informational purposes only.
It is not a medical device and should not be used for real diagnosis or treatment.
Always consult a certified medical professional for health-related issues.
