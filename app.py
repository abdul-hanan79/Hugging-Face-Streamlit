import streamlit as st
import torch
import io
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import wave
import numpy as np
# 1. Page Config & Title
st.set_page_config(page_title="Text To Speech | Abdul Hanan", page_icon=":microphone:", layout="centered")
st.title("Facebook MMS TTS (VitsModel) by Abdul Hanan ")

st.write("NumPy version:", np.__version__)
st.write("""
This app demonstrates the "facebook/mms-tts-eng" model, which uses a **forward pass** 
instead of `.generate()` to produce audio from text.
""")

# 2. Load Model & Tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")
    return tokenizer, model

tokenizer, model = load_model()

# 3. Sidebar Instructions
st.sidebar.header("How to Use:")
st.sidebar.write("""
1. Enter some text in English.
2. Click "Synthesize Speech".
3. Wait for processing and listen to the output audio.
""")

# 4. Form for Input
st.subheader("Enter Text to Synthesize")
with st.form("tts_form"):
    text_input = st.text_area("Text", value="Hello from MMS TTS!", height=100)
    submitted = st.form_submit_button("Synthesize Speech")

# 5. Synthesize & Play Audio
if submitted and text_input.strip():
    with st.spinner("Generating audio..."):
        # Prepare inputs
        inputs = tokenizer(text_input, return_tensors="pt")

        # MMS TTS uses forward pass, NOT `.generate()`
        with torch.no_grad():
            outputs = model(**inputs)
        
        waveforms = getattr(outputs, "waveform", None)
        if waveforms is None:
            st.error("No waveform found in model output. Check model documentation.")
        else:
            # waveforms is typically [1, num_samples]
            waveforms = waveforms.squeeze(0).cpu().numpy()

            # Write the waveform to an in-memory WAV using the standard library
            sample_rate = 16000  # MMS TTS typically uses 16kHz
            wav_buffer = io.BytesIO()

            # Convert float waveform to 16-bit integer PCM for WAV
            # scale from [-1,1] to int16
            pcm_16 = np.int16(waveforms * 32767)

            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)            # mono
                wf.setsampwidth(2)           # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_16.tobytes())

            wav_buffer.seek(0)

            st.balloons()
            st.success("Speech generated successfully!")
            st.audio(wav_buffer, format="audio/wav")

st.write("---")
st.write("**Model:** [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) • **Transformers** by [Hugging Face](https://github.com/huggingface/transformers)")
