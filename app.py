import streamlit as st
import torch
import io
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForTextToWaveform

# 1. Page Config & Title
st.set_page_config(page_title="Text To Speech | Abdul Hanan", page_icon=":microphone:", layout="centered")
st.title("Facebook MMS TTS (VitsModel) by Abdul Hanan ")

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

        # IMPORTANT: MMS TTS uses forward pass, NOT `.generate()`
        # The output object should contain the waveform
        with torch.no_grad():
            outputs = model(**inputs)
        
        # According to Hugging Face docs, outputs may have the key `.waveform`
        # Check the shape or dictionary to confirm:
        # waveforms = outputs.waveform  # or outputs["waveform"] if dict-like
        waveforms = getattr(outputs, "waveform", None)
        if waveforms is None:
            st.error("No waveform found in model output. Check model documentation.")
        else:
            # waveforms is typically [1, num_samples], so squeeze to 1D
            waveforms = waveforms.squeeze(0).cpu().numpy()

            # Write the waveform to an in-memory WAV
            sample_rate = 16000  # MMS TTS typically uses 16kHz
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, waveforms, sample_rate, format="WAV")
            wav_buffer.seek(0)
        
            st.balloons()
            st.success("Speech generated successfully!")
            st.audio(wav_buffer, format="audio/wav")

st.write("---")
st.write("**Model:** [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) • **Transformers** by [Hugging Face](https://github.com/huggingface/transformers)")

# transformers==4.26.1
