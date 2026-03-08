import streamlit as st
import os
import asyncio
import edge_tts
import sounddevice as sd
import scipy.io.wavfile as wavfile
import pygame
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough


# ================================
# CONFIG
# ================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Voice AI Assistant", layout="centered")


# ================================
# SPEECH TO TEXT
# ================================

class SpeechTranscriber:

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

    def record_audio(self, duration=4, fs=16000):

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        filename = "input.wav"
        wavfile.write(filename, fs, recording)

        return filename

    def transcribe(self, audio_path):

        with open(audio_path, "rb") as audio_file:

            transcript = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="en"
            )

        return transcript.text


# ================================
# RAG ENGINE
# ================================

class RAGEngine:

    def __init__(self):

        self.llm = ChatOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant"
        )

        texts = [
            "The warranty period for the X100 device is 2 years.",
            "To reset the device hold the power button for 10 seconds.",
            "The device supports WiFi and Bluetooth."
        ]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_texts(texts, embeddings)

        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful voice assistant.

Context:
{context}

Question:
{question}

Answer shortly."""
        )

        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, query):

        return self.chain.invoke(query)


# ================================
# TEXT TO SPEECH
# ================================

class TTSEngine:

    def __init__(self):
        self.voice = "en-US-AriaNeural"

    async def generate(self, text, file):

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(file)

    def speak(self, text):

        output_file = "response.mp3"

        asyncio.run(self.generate(text, output_file))

        return output_file


# ================================
# INIT SYSTEM
# ================================

if "stt" not in st.session_state:

    st.session_state.stt = SpeechTranscriber()
    st.session_state.rag = RAGEngine()
    st.session_state.tts = TTSEngine()


# ================================
# UI
# ================================

st.title("🤖 Voice AI Assistant")

st.write("Powered by **Groq (Llama 3.1) + Whisper + Edge-TTS**")

mode = st.radio("Input Mode", ["Voice", "Text"])


# ================================
# TEXT MODE
# ================================

if mode == "Text":

    user_text = st.text_input("Type your question")

    if st.button("Ask"):

        if user_text:

            with st.spinner("Thinking..."):

                response = st.session_state.rag.get_response(user_text)

                st.write("### Response")
                st.write(response)

                audio_file = st.session_state.tts.speak(response)

                st.audio(audio_file)


# ================================
# VOICE MODE
# ================================

if mode == "Voice":

    if st.button("🎤 Record Voice"):

        with st.spinner("Listening..."):

            audio_file = st.session_state.stt.record_audio()

        with st.spinner("Transcribing..."):

            text = st.session_state.stt.transcribe(audio_file)

        st.write("### You said")
        st.write(text)

        with st.spinner("Thinking..."):

            response = st.session_state.rag.get_response(text)

        st.write("### Assistant")
        st.write(response)

        audio_file = st.session_state.tts.speak(response)

        st.audio(audio_file)