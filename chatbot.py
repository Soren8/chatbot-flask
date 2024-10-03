import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, Response
import time
import threading
from queue import Queue, Empty
from collections import defaultdict
import requests
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ollama model name
MODEL_NAME = "dolphin3.1-8b-q6"

# Create Flask app
app = Flask(__name__)

# Session variables
sessions = defaultdict(
    lambda: {
        "history": [],
        "system_prompt": "You are a helpful AI assistant based on the Dolphin 3 8B model. Provide clear and concise answers to user queries.",
        "last_used": time.time(),
    }
)
SESSION_TIMEOUT = 3600  # 1 hour in seconds

response_lock = threading.Lock()

def clean_old_sessions():
    current_time = time.time()
    for session_id in list(sessions.keys()):
        if current_time - sessions[session_id]["last_used"] > SESSION_TIMEOUT:
            del sessions[session_id]

def generate_text_stream(prompt, system_prompt, model_name, session_history):
    url = "http://localhost:11434/api/generate"

    # Prepare the conversation history in the format expected by Ollama
    history_text = ""
    for user_input, assistant_response in session_history:
        history_text += f"### User:\n{user_input}\n\n### Assistant:\n{assistant_response}\n\n"
    # Append the latest user prompt
    history_text += f"### User:\n{prompt}\n\n### Assistant:\n"

    data = {
        "model": model_name,
        "prompt": history_text,
        "system": system_prompt,
        "stream": True,
    }

    # Start the streaming request to Ollama
    with requests.post(url, json=data, stream=True) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        yield json_response['response']
        else:
            yield f"\n[Error] Error: {response.status_code}, {response.text}"

@app.route("/")
def home():
    logger.info("Serving home page")
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400

    clean_old_sessions()

    session = sessions[session_id]
    session["last_used"] = time.time()

    user_message = request.json["message"]
    new_system_prompt = request.json.get("system_prompt")

    logger.info(
        f"Received chat request. Session: {session_id[:8]}... Message: {user_message[:50]}..."
    )

    if new_system_prompt is not None:
        logger.info(f"Updating system prompt to: {new_system_prompt[:50]}...")
        session["system_prompt"] = new_system_prompt

    if response_lock.locked():
        return jsonify(
            {
                "error": "A response is currently being generated. Please wait and try again."
            }
        ), 429

    def generate():
        with response_lock:
            # Start streaming the response
            stream = generate_text_stream(
                user_message, session["system_prompt"], MODEL_NAME, session["history"]
            )

            response_text = ""
            try:
                for chunk in stream:
                    response_text += chunk
                    yield chunk
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                yield "\n[Error] An error occurred during response generation."

            # Update conversation history after the full response is generated
            session["history"].append((user_message, response_text))
            logger.info(
                f"Chat response generated successfully. Length: {len(response_text)} characters"
            )

    return Response(generate(), mimetype="text/plain")

@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400

    if session_id in sessions:
        sessions[session_id]["history"] = []
        sessions[session_id][
            "system_prompt"
        ] = "You are a helpful AI assistant based on the Dolphin 3 8B model. Provide clear and concise answers to user queries."
        logger.info(f"Chat history has been reset for session {session_id[:8]}...")

    return jsonify({"status": "success", "message": "Chat history has been reset."})

@app.route("/health")
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    logger.info("Starting the Flask application")
    app.run(host="0.0.0.0", port=5000)
