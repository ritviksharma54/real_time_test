# Real-Time Voice AI with Gemini, Whisper & WebRTC

This project demonstrates a sophisticated, real-time, multilingual voice assistant built with a modern Python and JavaScript stack. It uses WebRTC to stream audio from a web client to a Python server for ultra-low latency, enabling natural, hands-free conversation with an AI.

The backend processes the audio stream using OpenAI's **Whisper** for transcription, generates conversational responses with **Google's Gemini Pro**, and synthesizes voice replies using **gTTS**. The entire system is orchestrated with Flask and Flask-SocketIO running on an ASGI server.

## 🌟 Core Features

*   **Real-Time Audio Streaming:** Utilizes **WebRTC** to stream microphone audio directly to the server, eliminating the need to upload audio files and significantly reducing latency.
*   **High-Accuracy Transcription:** Employs the **OpenAI Whisper** model for fast and accurate speech-to-text in multiple languages.
*   **Advanced Conversational AI:** Leverages **Google Gemini Pro** for intelligent, context-aware, and natural-sounding responses.
*   **Text-to-Speech Synthesis:** Converts the AI's text response back into speech using **gTTS (Google Text-to-Speech)**.
*   **Multilingual Support:** Currently configured for **English** and **Hindi**, easily extendable to other languages supported by Whisper and gTTS.
*   **Hands-Free Operation:** Features client-side **Voice Activity Detection (VAD)** to automatically detect when the user starts and stops speaking.
*   **Interactive UI:** A clean, responsive interface built with Bootstrap, featuring a chat history, status indicators, an audio visualizer, and a dynamic control button.
*   **Efficient Backend:** Built with **Flask** and **Flask-SocketIO** on an **ASGI** server (`Uvicorn`) to handle asynchronous operations like WebRTC and Socket.IO efficiently.
*   **Lazy Loading:** The Whisper model is loaded only on the first request to ensure a fast server startup time.

## ⚙️ Technology Stack

| Area | Technology |
| :--- | :--- |
| **Backend** | Python 3, Flask, Flask-SocketIO, Uvicorn (ASGI) |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Socket.IO Client, WebRTC (Browser API) |
| **AI/ML** | **Google Gemini API** (LLM), **OpenAI Whisper** (STT), **gTTS** (TTS), **PyTorch** (for Whisper) |
| **Real-Time** | **WebRTC** (`aiortc` on server, browser API on client), **Socket.IO** (for signaling and data) |
| **Audio** | `pydub` (for audio manipulation), `numpy` |
