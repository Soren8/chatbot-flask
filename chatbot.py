import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
import time
import threading
from queue import Queue, Empty
from collections import defaultdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Model path
model_path = r"meta-llama/Llama-3.2-3B-Instruct"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define context length
CONTEXT_LENGTH = 8192

# Load the tokenizer and model using Transformers
logger.info("Loading model and tokenizer. This may take a few moments...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()
logger.info("Model loaded successfully!")

# Create Flask app
app = Flask(__name__)

# Session variables
sessions = defaultdict(
    lambda: {
        "history": [],
        "system_prompt": "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, and always strive to provide accurate information.",
        "last_used": time.time(),
    }
)
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Approximate token counts (these are rough estimates, adjust as needed)
MAX_TOKENS = int(CONTEXT_LENGTH * 0.8)  # Leave some room for the new prompt and response
TOKENS_PER_MESSAGE = 4  # Rough estimate for prompt formatting
APPROXIMATE_TOKENS_PER_WORD = 1.3

response_lock = threading.Lock()

def estimate_tokens(text):
    return int(len(tokenizer.encode(text, add_special_tokens=False)))

def manage_conversation_history(history, new_prompt, new_response):
    # Add new exchange to history
    history.append((new_prompt, new_response))

    # Calculate total tokens
    total_tokens = sum(
        estimate_tokens(prompt)
        + estimate_tokens(response)
        + TOKENS_PER_MESSAGE
        for prompt, response in history
    )

    # Remove oldest exchanges if we exceed the limit
    while total_tokens > MAX_TOKENS and history:
        removed_prompt, removed_response = history.pop(0)
        total_tokens -= (
            estimate_tokens(removed_prompt)
            + estimate_tokens(removed_response)
            + TOKENS_PER_MESSAGE
        )
        logger.info(
            f"Removed oldest exchange to maintain context length. Current estimated tokens: {total_tokens}"
        )

    return history

class StopOnUserPrompt(StoppingCriteria):
    def __init__(self, user_prompt_str, tokenizer):
        super().__init__()
        self.user_prompt_str = user_prompt_str
        self.user_prompt_ids = tokenizer.encode(
            user_prompt_str, add_special_tokens=False
        )
        self.buffer = []

    def __call__(self, input_ids, scores, **kwargs):
        # Append the latest tokens
        self.buffer.extend(input_ids[0, -len(self.user_prompt_ids) :].tolist())

        # Check if the buffer ends with the user prompt
        if self.buffer[-len(self.user_prompt_ids) :] == self.user_prompt_ids:
            return True
        return False

def generate_and_stream(prompt, system_prompt, history):
    # Prepare the conversation history
    history_text = "".join(
        [
            f"### User:\n{p}\n\n### Assistant:\n{r}\n\n"
            for p, r in history
        ]
    )

    # Add instruct-style prompt tags with the system prompt
    formatted_prompt = (
        f"### System:\n{system_prompt}\n\n{history_text}### User:\n{prompt}\n\n### Assistant:\n"
    )

    logger.info(f"Generating response for prompt: {prompt[:50]}...")
    logger.info(f"Using system prompt: {system_prompt[:50]}...")

    # Tokenize the input prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Initialize the streamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Define custom stopping criteria
    stopping_criteria = StoppingCriteriaList(
        [StopOnUserPrompt("### User:", tokenizer)]
    )

    # Define generation parameters
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": int(CONTEXT_LENGTH * 0.5),
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "stopping_criteria": stopping_criteria,
        "streamer": streamer,
    }

    # Start generation in a separate thread
    generation_thread = threading.Thread(
        target=model.generate, kwargs=generation_kwargs
    )
    generation_thread.start()

    # Yield the generated text as it becomes available
    for new_text in streamer:
        yield new_text

    # Wait for the generation thread to complete
    generation_thread.join()

@app.route("/")
def home():
    logger.info("Serving home page")
    return render_template("chat.html")

def clean_old_sessions():
    current_time = time.time()
    for session_id in list(sessions.keys()):
        if current_time - sessions[session_id]["last_used"] > SESSION_TIMEOUT:
            del sessions[session_id]

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

    if new_system_prompt:
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
            stream = generate_and_stream(
                user_message, session["system_prompt"], session["history"]
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
            session["history"] = manage_conversation_history(
                session["history"], user_message, response_text
            )
            logger.info(
                f"Chat response generated successfully. Length: {len(response_text)} characters"
            )

    return Response(generate(), mimetype="text/plain")

@app.route("/health")
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "healthy"}), 200

@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400

    if session_id in sessions:
        sessions[session_id]["history"] = []
        sessions[session_id][
            "system_prompt"
        ] = "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, and always strive to provide accurate information."
        logger.info(f"Chat history has been reset for session {session_id[:8]}...")

    return jsonify({"status": "success", "message": "Chat history has been reset."})

if __name__ == "__main__":
    logger.info("Starting the Flask application")
    app.run(host="0.0.0.0", port=5000)
