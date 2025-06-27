import os
import base64
import threading
import io
import numpy as np
import torch
import time
from pydub import AudioSegment
from dotenv import load_dotenv

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import whisper

load_dotenv() # Load environment variables from .env file

# --- gTTS specific import ---
try:
    from gtts import gTTS, gTTSError
    GTTS_AVAILABLE = True
    print("✅ gTTS library imported successfully.")
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️ gTTS library not found. Server-side TTS with gTTS will be disabled.")
    print("   Install it with: pip install gTTS")

# Initialize Flask & SocketIO
app = Flask("voice_app")
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your-default-secret-key-CHANGE-ME')
socketio_app = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Configure Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_model_global_instance = None # We will create instances with specific system prompts later

# Gemini Generation Config and Safety Settings (can be global)
generation_config = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_output_tokens": 1024}
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Original generic system instruction (for fallback or reference)
original_generic_system_instruction = """You are a abusive and rude multilingual voice assistant.
Your primary goal is to be mean to the user effectively.
Keep your responses conversational, concise (usually 1-3 sentences), and natural-sounding for speech.
Refer to previous parts of the conversation if relevant to provide context and continuity.
Do not use markdown in your responses.
"""

# Template for language-specific system instruction
system_instruction_template = """You are a friendly and engaging voice assistant.
The user is speaking in {user_language_name} (language code: {user_language_code}).
You MUST respond *only* in {user_language_name} ({user_language_code}).
Keep your responses conversational, concise (usually 1-3 sentences), and natural-sounding for speech.
Refer to previous parts of the conversation if relevant to provide context and continuity.
Do not use markdown in your responses.
"""

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # We don't create a global gemini_model with a fixed system instruction here anymore.
    # It will be created on-demand with the correct language context.
    print("✅ Gemini API Key configured. Model instances will be created per session/language.")
    gemini_model_global_instance = True # Just a flag to indicate API key is set
else:
    print("⚠️ GEMINI_API_KEY not found. AI features will use fallback responses.")


# --- Whisper STT Setup ---
whisper_model = None
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium") # Changed default to medium
device_whisper = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print(f"Attempting to load Whisper model '{WHISPER_MODEL_SIZE}' on device '{device_whisper}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device_whisper)
    print(f"✅ Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully on '{device_whisper}'.")
except Exception as e:
    print(f"❌ Error loading Whisper model '{WHISPER_MODEL_SIZE}': {e}")
    whisper_model = None


# --- gTTS Synthesis Function ---
def synthesize_speech_with_gtts(text, lang_code_iso):
    if not GTTS_AVAILABLE:
        print("gTTS library not available for synthesis.")
        return None, None

    base_lang_code = lang_code_iso.split('-')[0].lower()
    print(f"Synthesizing with gTTS for lang='{base_lang_code}'...")
    try:
        tts = gTTS(text=text, lang=base_lang_code, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue(), "audio/mpeg"
    except gTTSError as e_gtts:
        print(f"❌ Error during gTTS synthesis for '{base_lang_code}': {e_gtts}")
        if base_lang_code != 'en': # Try English as a last resort if primary fails
            print(f"⚠️ Retrying gTTS with 'en' due to error with '{base_lang_code}'.")
            try:
                tts_en = gTTS(text=text, lang='en', slow=False)
                mp3_fp_en = io.BytesIO()
                tts_en.write_to_fp(mp3_fp_en)
                mp3_fp_en.seek(0)
                return mp3_fp_en.getvalue(), "audio/mpeg"
            except Exception as e_en:
                print(f"❌ Error during gTTS 'en' fallback synthesis: {e_en}")
        return None, None
    except Exception as e_synth:
        print(f"❌ Unexpected error during gTTS synthesis for '{base_lang_code}': {e_synth}")
        return None, None

# --- Conversation State Management ---
# chat_sessions will store instances of GenerativeModel.start_chat()
chat_sessions = {} # session_id -> { "chat_object": chat, "_current_lang": "en" }

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test-gemini", methods=["GET"])
def test_gemini_route():
    if not gemini_model_global_instance: # Check if API key was configured
        return jsonify({"error": "Gemini API not configured"}), 500

    test_lang_code = request.args.get('lang', 'en')
    if test_lang_code not in ['en', 'hi']:
        test_lang_code = 'en' # Default to English if invalid lang for test

    language_name = "English" if test_lang_code == "en" else "Hindi"
    test_system_instruction = system_instruction_template.format(
        user_language_name=language_name,
        user_language_code=test_lang_code
    )
    test_prompt_text = "Say hello for an API test!"
    if test_lang_code == 'hi':
        test_prompt_text = "एपीआई परीक्षण के लिए नमस्ते कहो!"


    try:
        # Create a temporary model instance for the test with the specific system instruction
        temp_test_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=test_system_instruction
        )
        response = temp_test_model.generate_content(test_prompt_text)
        ai_response_text = response.text
        
        response_audio_base64 = None
        tts_mime_type = None

        if GTTS_AVAILABLE:
            # TTS should be in the language of the response (test_lang_code)
            audio_bytes, mime_type = synthesize_speech_with_gtts(ai_response_text, test_lang_code)
            if audio_bytes:
                response_audio_base64 = f"data:{mime_type};base64," + base64.b64encode(audio_bytes).decode('utf-8')
                tts_mime_type = mime_type
        
        return jsonify({
            "success": True,
            "response_text": ai_response_text,
            "response_audio_base64": response_audio_base64,
            "audio_mime_type": tts_mime_type,
            "response_language": test_lang_code, # Language of the response
            "model_name": "gemini-1.5-flash"
        })
    except Exception as e:
        print(f"❌ Error during /test-gemini: {e}")
        return jsonify({"error": str(e), "lang_used": test_lang_code}), 500

# --- Main Audio Processing Function ---
def process_audio_with_gemini(audio_data_base64, user_id, target_lang_code):
    if not whisper_model:
        return {"success": False, "error": "Speech recognition model not available.", "user_id": user_id, "target_lang": target_lang_code}

    if target_lang_code not in ["en", "hi"]:
        print(f"⚠️ Unsupported target_lang_code '{target_lang_code}' received. Returning error.")
        return {"success": False, "error": f"Unsupported language '{target_lang_code}'. Please select English or Hindi.", "user_id": user_id, "target_lang": target_lang_code}

    try:
        header, encoded = audio_data_base64.split(",", 1)
        audio_bytes = base64.b64decode(encoded)
        
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        max_val = np.iinfo(audio_segment.array_type).max if audio_segment.sample_width > 0 else 32768.0
        if max_val == 0: max_val = 32768.0 
        samples /= max_val
        
        print(f"🎤 Transcribing with Whisper for language: '{target_lang_code}' for user {user_id}")
        transcription_result = whisper_model.transcribe(
            samples,
            language=target_lang_code,
            fp16=(device_whisper=="cuda")
        )
        user_text = transcription_result["text"].strip()
        actual_detected_by_whisper = transcription_result["language"] 
        print(f"🗣️ Whisper STT ({user_id}): Target='{target_lang_code}', Detected='{actual_detected_by_whisper}', Text='{user_text}'")

        if not user_text:
             return {"success": True, "transcription": "", 
                     "detected_language": target_lang_code, "response_text": "", 
                     "response_language": target_lang_code, "user_id": user_id,
                     "info": "No speech detected."}

        ai_response_text = "Sorry, I am unable to process that right now."
        if target_lang_code == 'hi':
            ai_response_text = "क्षमा करें, मैं अभी यह संसाधित नहीं कर सकता।"
        
        ai_response_language_for_llm_and_tts = target_lang_code

        if gemini_model_global_instance: # If API key is configured
            current_chat_session_data = chat_sessions.get(user_id)
            chat_object = None

            language_name = "English" if target_lang_code == "en" else "Hindi"
            current_system_instruction = system_instruction_template.format(
                user_language_name=language_name,
                user_language_code=target_lang_code
            )

            # Check if chat needs re-initialization due to language change or first time
            if not current_chat_session_data or current_chat_session_data.get("_current_lang") != target_lang_code:
                print(f"🔄 Initializing/Resetting Gemini chat for {user_id} with lang {target_lang_code}")
                temp_gemini_model_for_chat = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    system_instruction=current_system_instruction
                )
                chat_object = temp_gemini_model_for_chat.start_chat(history=[])
                chat_sessions[user_id] = {"chat_object": chat_object, "_current_lang": target_lang_code}
            else:
                chat_object = current_chat_session_data["chat_object"]

            if chat_object:
                try:
                    gemini_response = chat_object.send_message(user_text)
                    ai_response_text = gemini_response.text
                    print(f"🤖 Gemini ({user_id}, lang_context: {target_lang_code}): \"{ai_response_text}\"")
                except Exception as e_gemini:
                    print(f"❌ Gemini API error for {user_id}: {e_gemini}")
                    ai_response_text = "I encountered an issue with the AI model."
                    if target_lang_code == "hi":
                       ai_response_text = "मुझे AI मॉडल के साथ एक समस्या आई।"
            else: # Should not happen if logic above is correct
                print(f"⚠️ Chat object not found for {user_id}. This is unexpected.")
        
        elif not gemini_model_global_instance:
            ai_response_text = get_fallback_response(user_text.lower(), target_lang_code)
        
        response_audio_base64 = None
        tts_synthesis_error = None
        audio_mime_type = None

        if GTTS_AVAILABLE and ai_response_text:
            audio_bytes, mime_type = synthesize_speech_with_gtts(ai_response_text, ai_response_language_for_llm_and_tts)
            if audio_bytes:
                response_audio_base64 = f"data:{mime_type};base64," + base64.b64encode(audio_bytes).decode('utf-8')
                audio_mime_type = mime_type
                print(f"🎤 gTTS synthesized audio for '{ai_response_language_for_llm_and_tts}'. Type: {mime_type}")
            else:
                tts_synthesis_error = f"gTTS synthesis failed for language: {ai_response_language_for_llm_and_tts}"
        
        return_payload = {
            "success": True, "transcription": user_text, 
            "detected_language": ai_response_language_for_llm_and_tts,
            "response_text": ai_response_text, 
            "response_language": ai_response_language_for_llm_and_tts,
            "user_id": user_id, "target_lang": target_lang_code
        }
        if response_audio_base64:
            return_payload["response_audio_base64"] = response_audio_base64
            return_payload["audio_mime_type"] = audio_mime_type
        if tts_synthesis_error and not response_audio_base64:
            return_payload["tts_error"] = tts_synthesis_error
        
        return return_payload

    except Exception as e:
        print(f"❌ Unexpected error in process_audio_with_gemini for {user_id} with target_lang {target_lang_code}: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": "Internal server error during audio processing.", "user_id": user_id, "target_lang": target_lang_code}

def get_fallback_response(text_query, lang_code="en"):
    if lang_code == "hi": return "क्षमा करें, मैं अभी आपकी मदद नहीं कर सकता।"
    return "I'm sorry, I can't help with that right now."

# --- SocketIO Event Handlers ---
@socketio_app.on('connect')
def on_connect(auth=None):
    user_id = request.sid
    print(f"✅ User connected: {user_id}")
    # Chat session will be initialized on first audio_data with language context
    emit('connection_status', {'status': 'connected', 'user_id': user_id, 'model_ready': bool(gemini_model_global_instance)})

@socketio_app.on('disconnect')
def on_disconnect():
    user_id = request.sid
    print(f"❌ User disconnected: {user_id}")
    if user_id in chat_sessions:
        del chat_sessions[user_id]
        print(f"🗑️ Chat session removed for {user_id}")

@socketio_app.on('audio_data')
def handle_audio_data(data):
    user_id = request.sid
    audio_data_base64 = data.get('audio')
    target_lang_code = data.get('target_lang', 'en') 

    if not audio_data_base64:
        emit('audio_response', {'success': False, 'error': 'No audio data received', 'user_id': user_id, "target_lang": target_lang_code})
        return
    
    if not whisper_model:
        emit('audio_response', {'success': False, 'error': 'Speech recognition service is not available.', 'user_id': user_id, "target_lang": target_lang_code})
        return

    def process_audio_thread_target():
        result = process_audio_with_gemini(audio_data_base64, user_id, target_lang_code)
        socketio_app.emit('audio_response', result, room=user_id)

    thread = threading.Thread(target=process_audio_thread_target)
    thread.daemon = True
    thread.start()

@socketio_app.on('clear_conversation')
def handle_clear_conversation():
    user_id = request.sid
    if user_id in chat_sessions:
        # Simply deleting the session will cause it to be re-initialized
        # with the correct language context on the next audio_data event.
        del chat_sessions[user_id]
        print(f"🔄 Conversation data cleared for {user_id}. Chat will re-initialize on next input.")
        emit('conversation_cleared', {'status': 'success', 'message': 'Conversation history cleared.'})
    else:
        emit('conversation_cleared', {'status': 'success', 'message': 'No active conversation to clear.'})


# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Real-time Voice AI Server...")
    if not GEMINI_API_KEY: print("🔴 CRITICAL: GEMINI_API_KEY not set.")
    elif not gemini_model_global_instance: print("🟠 WARNING: Gemini model init (flag) not set, though API key might be present.") # Should not happen if key is set
    if not whisper_model: print("🔴 CRITICAL: Whisper STT model FAILED to load.")
    
    if not GTTS_AVAILABLE:
        print("🔴 gTTS library is not installed. Server-side TTS is disabled.")
    else:
        print("✅ gTTS is configured for server-side synthesis.")
    
    socketio_app.run(app, host="0.0.0.0", port=8116, debug=True, use_reloader=False, allow_unsafe_werkzeug=True if os.environ.get("WERKZEUG_DEBUG_PIN") == "off" else False)
