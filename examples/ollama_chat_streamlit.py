"""
Streamlit chat interface with Ollama + TTS streaming.

Usage:
    streamlit run examples/ollama_chat_streamlit.py

Requires:
    - Ollama running on localhost:11434
    - TTS backend running on localhost:8000 (python examples/tts_backend.py)
"""

import streamlit as st
import requests
import json
import base64
import numpy as np
import soundfile as sf
from io import BytesIO

OLLAMA_BASE = "http://localhost:11434"
TTS_BASE = "http://localhost:8000"


def get_ollama_models():
    """Fetch list of available models from Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return sorted([m["name"] for m in models])
    except Exception as e:
        st.error(f"Failed to connect to Ollama: {e}")
    return []


def check_tts_backend():
    """Check if TTS backend is available."""
    try:
        resp = requests.get(f"{TTS_BASE}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def synthesize_audio(text, speed=1.0, pitch=1.0, energy=1.0):
    """Call TTS backend to synthesize audio and decode to numpy array."""
    try:
        resp = requests.post(
            f"{TTS_BASE}/synthesize",
            json={
                "text": text,
                "speed": speed,
                "pitch": pitch,
                "energy": energy
            },
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            # Decode base64 WAV bytes
            wav_bytes = base64.b64decode(data["audio"])
            # Load WAV back into numpy array
            wav_io = BytesIO(wav_bytes)
            audio_array, sr = sf.read(wav_io)
            return audio_array, data["duration"]
    except Exception as e:
        st.error(f"TTS synthesis failed: {e}")
    return None, None


def stream_ollama_response(model, message):
    """Stream response from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": message,
                "stream": True,
            },
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                text = chunk.get("response", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue
    except Exception as e:
        st.error(f"Ollama error: {e}")


# UI Setup
st.set_page_config(page_title="Ollama Chat with TTS", layout="wide")
st.title("🎤 Ollama Chat with TTS Audio")

# Check backends
ollama_ok = True
tts_ok = check_tts_backend()

col1, col2 = st.columns(2)
with col1:
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
    except:
        st.error("❌ Ollama not running on localhost:11434")
        ollama_ok = False

with col2:
    if tts_ok:
        st.success("✅ TTS backend ready")
    else:
        st.error("❌ TTS backend not running (python tts_backend.py)")

if not (ollama_ok and tts_ok):
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    models = get_ollama_models()
    if not models:
        st.error("No models found in Ollama")
        st.stop()

    model = st.selectbox("Model", models, index=0)

    st.subheader("TTS Controls")
    speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)
    energy = st.slider("Energy", 0.5, 2.0, 1.0, 0.1)

# Chat interface
st.subheader("Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/wav")

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Stream response from Ollama
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        audio_placeholder = st.empty()
        status_placeholder = st.empty()

        full_response = ""
        token_buffer = ""
        all_audio_chunks = []
        combined_audio = BytesIO()

        # Write WAV header manually for proper concatenation
        import struct
        wav_header_written = False

        def write_wav_header(f, num_channels=1, sample_rate=22050, num_samples=0):
            """Write WAV file header"""
            byte_rate = sample_rate * num_channels * 2
            block_align = num_channels * 2
            subchunk2_size = num_samples * num_channels * 2
            chunk_size = 36 + subchunk2_size

            f.write(b'RIFF')
            f.write(struct.pack('<I', chunk_size))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # subchunk1 size
            f.write(struct.pack('<H', 1))   # audio format (PCM)
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', 16))  # bits per sample
            f.write(b'data')
            f.write(struct.pack('<I', subchunk2_size))

        for token in stream_ollama_response(model, user_input):
            full_response += token
            token_buffer += token
            response_placeholder.write(full_response)

            # Generate audio only at sentence boundaries (., !, ?)
            # Look for complete sentences in the buffer
            while True:
                sentence_end = -1
                for i, char in enumerate(token_buffer):
                    if char in '.!?':
                        sentence_end = i
                        break

                if sentence_end == -1:
                    # No complete sentence yet, wait for more tokens
                    break

                # Extract complete sentence (including punctuation)
                sentence = token_buffer[:sentence_end + 1].strip()
                token_buffer = token_buffer[sentence_end + 1:].strip()

                if sentence:
                    status_placeholder.info(f"🎤 Generating: '{sentence[:60]}...'")
                    wav_data, duration = synthesize_audio(
                        sentence,
                        speed=speed,
                        pitch=pitch,
                        energy=energy
                    )
                    if wav_data is not None:
                        # Normalize audio to prevent clipping
                        max_val = np.abs(wav_data).max()
                        if max_val > 1.0:
                            wav_data = wav_data / max_val * 0.95
                        all_audio_chunks.append(wav_data)
                        status_placeholder.success(f"✓ {duration:.2f}s")

        # Generate audio for remaining text (if any)
        if token_buffer.strip():
            status_placeholder.info(f"🎤 Generating final chunk...")
            wav_data, duration = synthesize_audio(
                token_buffer.strip(),
                speed=speed,
                pitch=pitch,
                energy=energy
            )
            if wav_data is not None:
                # Normalize audio
                max_val = np.abs(wav_data).max()
                if max_val > 1.0:
                    wav_data = wav_data / max_val * 0.95
                all_audio_chunks.append(wav_data)

        # Combine all audio chunks into single WAV file
        if all_audio_chunks:
            status_placeholder.info("🔄 Combining audio chunks...")

            import soundfile as sf
            import numpy as np

            # Concatenate all audio arrays
            full_audio = np.concatenate(all_audio_chunks)

            # Write to BytesIO as proper WAV
            combined_audio = BytesIO()
            sf.write(combined_audio, full_audio, 22050, format='WAV')
            combined_audio.seek(0)

            # Display with autoplay via HTML
            audio_bytes = combined_audio.getvalue()
            audio_b64 = base64.b64encode(audio_bytes).decode()

            audio_html = f"""
            <audio controls autoplay style="width: 100%; margin-top: 10px;">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

            # Also show the audio player for replay
            st.write("**🔊 Replay full response:**")
            st.audio(combined_audio, format="audio/wav")

            status_placeholder.success(f"✅ Complete! Total: {len(full_audio)/22050:.2f}s")

            # Save to history
            combined_audio.seek(0)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "audio": combined_audio.getvalue()
            })
