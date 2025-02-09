import streamlit as st
import asyncio
import os
import requests
import subprocess
import mimetypes
from deepgram import DeepgramClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from deep_translator import GoogleTranslator
import whisper

import streamlit as st

# Load API keys from secrets
GROQ_API_KEY = st.secrets["general"]["GROQ_API_KEY"]
DEEPGRAM_API_KEY = st.secrets["general"]["DEEPGRAM_API_KEY"]

if not GROQ_API_KEY or not DEEPGRAM_API_KEY:
    st.error("❌ Missing API keys. Please set them in the script.")
    st.stop()

# Function to translate text
def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# AI Roleplay & Language Tutor Processor
class LanguageModelProcessor:
    def __init__(self, role="Tutor"):
        system_prompt = f"You are a {role} helping the user practice language. Maintain a conversational flow and correct mistakes."
        self.llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def process(self, text):
        response = self.conversation.invoke({"text": text})
        return response['text']

# Text-to-Speech Class
class TextToSpeech:
    MODEL_NAME = "aura-helios-en"
    
    def __init__(self):
        self.process = None

    def speak(self, text, target_language):
        translated_text = translate_text(text, target_language)
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&encoding=linear16"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        payload = {"text": translated_text}
        
        player_command = ["ffplay", "-autoexit", "-nodisp", "-"]
        self.process = subprocess.Popen(player_command, stdin=subprocess.PIPE)
        
        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk and self.process:
                    self.process.stdin.write(chunk)
                    self.process.stdin.flush()
        
        if self.process:
            self.process.stdin.close()
            self.process.wait()
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process = None

# Function to transcribe audio using Whisper
async def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# Streamlit UI
st.title("🌍 AI-Powered Language Learning")
st.sidebar.header("Settings")
target_language = st.sidebar.selectbox("Select Target Language", ["en", "es", "fr", "de", "zh", "ar", "hi"])  # Added Hindi (hi)
role = st.sidebar.selectbox("Choose AI Role", ["Tutor", "Travel Agent", "Waiter", "Job Interviewer"])

# Roleplay AI
ai_tutor = LanguageModelProcessor(role=role)

tts = TextToSpeech()

# User Input
txt_input = st.text_input("Type or Speak a Sentence:")
if txt_input:
    response = ai_tutor.process(txt_input)
    translated_response = translate_text(response, target_language)
    st.write("**AI Response:**", response)
    st.write("**Translated AI Response:**", translated_response)

    if st.button("🔊 Hear AI Response"):
        tts.speak(translated_response, target_language)
    if st.button("⏹️ Stop AI Response"):
        tts.stop()

# File Upload for Speech Recognition
uploaded_file = st.file_uploader("Upload an MP3/WAV file", type=["mp3", "wav"])
if uploaded_file:
    file_extension = mimetypes.guess_extension(uploaded_file.type)
    
    if not file_extension:
        st.error("❌ Unsupported file format.")
    else:
        file_path = f"temp_audio{file_extension}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        transcript = asyncio.run(transcribe_audio(file_path))
        os.remove(file_path)
        
        st.write("**You Said:**", transcript)
        translated_text = translate_text(transcript, target_language)
        st.write("**Translation:**", translated_text)
        
        ai_feedback = ai_tutor.process(transcript)
        translated_feedback = translate_text(ai_feedback, target_language)
        st.write("**AI Feedback:**", ai_feedback)
        st.write("**Translated AI Feedback:**", translated_feedback)
        
        if st.button("🔊 Hear AI Feedback"):
            tts.speak(translated_feedback, target_language)
        if st.button("⏹️ Stop AI Feedback"):
            tts.stop()
