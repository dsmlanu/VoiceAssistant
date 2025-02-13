import os
import whisper
import google.cloud.dialogflow_v2 as dialogflow
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import streamlit as st

# Load models
whisper_model = whisper.load_model("base")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")

# Set up Dialogflow authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dialogflow_key.json"

def speech_to_text(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result["text"]

def detect_intent(text, session_id="12345"):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path("your-dialogflow-project-id", session_id)

    text_input = dialogflow.TextInput(text=text, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)
    response = session_client.detect_intent(session=session, query_input=query_input)

    return response.query_result.fulfillment_text

def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)

# Streamlit Web App
st.title("🎙️ LLaMA 2 Voice Assistant")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.text("Converting Speech to Text...")
    text = speech_to_text("temp_audio.wav")
    st.write(f"🗣️ You said: {text}")

    st.text("Processing with Dialogflow...")
    intent_response = detect_intent(text)

    if "default response" in intent_response.lower():
        st.text("Generating AI Response...")
        ai_response = generate_response(text)
    else:
        ai_response = intent_response

    st.write(f"🤖 AI: {ai_response}")

    st.text("Converting AI response to Speech...")
    text_to_speech(ai_response, "response.mp3")
    st.audio("response.mp3")
