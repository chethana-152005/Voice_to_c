import streamlit as st
import os
import asyncio
import edge_tts
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in .env file.") 

# ==========================================
# 2. CORE LOGIC CLASSES
# ==========================================

# --- TTS Function (Stand-alone for async) ---
async def generate_tts_audio(text, output_file):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)

# --- RAG Engine Class ---
class RAGEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant", 
            temperature=0
        )
        
        # Demo Knowledge Base
        texts = [
            "The warranty period for the X100 device is 2 years.",
            "To reset the device, hold the power button for 10 seconds.",
            "The device supports both WiFi and Bluetooth connections."
        ]
        
        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(texts, embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful voice assistant.\n"
            "Use the following context if it helps:\n{context}\n\n"
            "If the context is not relevant, answer using your general knowledge.\n"
            "Question: {question}\n"
            "Answer concisely:"
        )
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )

    def get_response(self, query):
        return self.chain.invoke(query)

# --- STT Client ---
stt_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ==========================================
# 3. STREAMLIT APP
# ==========================================

def main():
    st.set_page_config(page_title="Voice AI Assistant", page_icon="🤖")
    
    st.title("🤖 Voice AI Assistant")
    st.markdown("### Powered by Groq (Llama 3.1) & Edge-TTS")

    # Initialize Session State for Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Cache the RAG Engine so it doesn't reload every interaction
    @st.cache_resource
    def load_rag_engine():
        with st.spinner("Loading AI Models..."):
            return RAGEngine()

    rag_engine = load_rag_engine()

    # --- Sidebar for Settings ---
    with st.sidebar:
        st.header("Settings")
        # Option to type or speak
        input_mode = st.radio("Input Mode", ["Voice 🎤", "Text ⌨️"], horizontal=True)
        st.markdown("---")
        st.write("### Chat History")
        # Display previous messages in sidebar
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.write(f"👤: {msg['content']}")
            else:
                st.write(f"🤖: {msg['content']}")

    # --- Main Interaction Area ---

    # 1. INPUT HANDLING
    user_input = None

    if input_mode == "Text ⌨️":
        user_input = st.chat_input("Type your message here...")
        
    else: # Voice Mode
        st.info("Click the microphone icon below to record.")
        # Streamlit Audio Widget
        audio_bytes = st.audio_input("Record your query")
        
        if audio_bytes:
            # Save temp file for Groq Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes.getvalue())
                tmp_path = tmp.name
            
            # Transcribe
            with st.spinner("Transcribing..."):
                try:
                    with open(tmp_path, "rb") as audio_file:
                        transcript = stt_client.audio.transcriptions.create(
                            file=audio_file,
                            model="whisper-large-v3",
                            language="en"
                        )
                    user_input = transcript.text
                    st.success(f"Heard: {user_input}")
                except Exception as e:
                    st.error(f"STT Error: {e}")
                finally:
                    os.remove(tmp_path)

    # 2. PROCESS & RESPOND
    if user_input:
        # Display User Message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate Response
        with st.spinner("Thinking..."):
            try:
                bot_response = rag_engine.get_response(user_input)
            except Exception as e:
                bot_response = f"Error generating response: {e}"

        # Display Bot Text Response
        st.chat_message("assistant").markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Generate TTS Audio
        with st.spinner("Generating Voice..."):
            tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            try:
                asyncio.run(generate_tts_audio(bot_response, tts_file.name))
                
                # Play Audio in Streamlit
                st.audio(tts_file.name, format="audio/mp3")
            except Exception as e:
                st.error(f"TTS Error: {e}")

if __name__ == "__main__":
    main()