import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
from functools import wraps
import time
import threading
from queue import Queue, Empty
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define context length
CONTEXT_LENGTH = 8192

# Initialize the model with the provided path and increased context length
model_path = r"F:\lmstudio\cognitivecomputations\dolphin-2.9-llama3-8b-gguf\dolphin-2.9-llama3-8b-q3_K_M.gguf"

# Load the tokenizer and model using Transformers
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

# Create Flask app
app = Flask(__name__)

# Session variables
sessions = defaultdict(lambda: {"history": [], "system_prompt": "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, and always strive to provide accurate information.", "last_used": time.time()})
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Approximate token counts (these are rough estimates, adjust as needed)
MAX_TOKENS = int(CONTEXT_LENGTH * 0.8) # Leave some room for the new prompt and response
TOKENS_PER_MESSAGE = 4  # Rough estimate for "### Human: " and "### Assistant: "
APPROXIMATE_TOKENS_PER_WORD = 1.3

response_lock = threading.Lock()

def estimate_tokens(text):
    return int(len(text.split()) * APPROXIMATE_TOKENS_PER_WORD)

def manage_conversation_history(history, new_prompt, new_response):
    # Add new exchange to history
    history.append((new_prompt, new_response))

    # Calculate total tokens
    total_tokens = sum(estimate_tokens(prompt) + estimate_tokens(response) + TOKENS_PER_MESSAGE
                       for prompt, response in history)

    # Remove oldest exchanges if we exceed the limit
    while total_tokens > MAX_TOKENS and history:
        removed_prompt, removed_response = history.pop(0)
        total_tokens -= (estimate_tokens(removed_prompt) + estimate_tokens(removed_response) + TOKENS_PER_MESSAGE)
        logger.info(f"Removed oldest exchange to maintain context length. Current estimated tokens: {total_tokens}")

    return history

def retry_on_exception(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_on_exception()
def generate_response(prompt, system_prompt, history):
    history_text = "\n".join([f"### Human: {p}\n\n### Assistant: {r}" for p, r in history])

    full_prompt = f"""{system_prompt}

{history_text}

### Human: {prompt}

### Assistant: """

    logger.info(f"Generating response for prompt: {prompt[:50]}...")
    logger.info(f"Using system prompt: {system_prompt[:50]}...")

    def generate():
        nonlocal full_prompt
        # Encode the full prompt
        input_ids = tokenizer.encode(full_prompt, return_tensors='pt')
        # Move tensors to the appropriate device
        input_ids = input_ids.to(model.device)
        # Generate the response
        output_ids = model.generate(
            input_ids,
            max_new_tokens=int(CONTEXT_LENGTH * 0.5),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Get the generated tokens (excluding the prompt)
        generated_tokens = output_ids[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Optionally, truncate at stop sequences
        stop_sequence = "### Human:"
        if stop_sequence in response:
            response = response.split(stop_sequence)[0]
        return response.strip()

    queue = Queue()
    thread = threading.Thread(target=lambda q, fn: q.put(fn()), args=(queue, generate))
    thread.start()

    try:
        response = queue.get(block=True, timeout=30)  # 30 second timeout
        logger.info(f"Response generated successfully. Length: {len(response)} characters")

        # Update conversation history
        history = manage_conversation_history(history, prompt, response)

        return response, history
    except Empty:
        logger.error("Response generation timed out after 30 seconds")
        raise Exception("Response generation timed out")
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
    finally:
        thread.join(timeout=1)

@app.route('/')
def home():
    logger.info("Serving home page")
    return render_template('chat.html')

def clean_old_sessions():
    current_time = time.time()
    for session_id in list(sessions.keys()):
        if current_time - sessions[session_id]["last_used"] > SESSION_TIMEOUT:
            del sessions[session_id]

@app.route('/chat', methods=['POST'])
def chat():
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    clean_old_sessions()

    session = sessions[session_id]
    session["last_used"] = time.time()

    user_message = request.json['message']
    new_system_prompt = request.json.get('system_prompt')

    logger.info(f"Received chat request. Session: {session_id[:8]}... Message: {user_message[:50]}...")

    if new_system_prompt:
        logger.info(f"Updating system prompt to: {new_system_prompt[:50]}...")
        session["system_prompt"] = new_system_prompt

    if response_lock.locked():
        return jsonify({'error': 'A response is currently being generated. Please wait and try again.'}), 429

    try:
        with response_lock:
            ai_response, session["history"] = generate_response(user_message, session["system_prompt"], session["history"])
            logger.info(f"Chat response generated successfully. Length: {len(ai_response)} characters")
            logger.info("Response preview:")
            logger.info(ai_response[:200] + "..." if len(ai_response) > 200 else ai_response)

        return jsonify({'response': ai_response, 'system_prompt': session["system_prompt"]})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        if "timed out" in str(e).lower():
            return jsonify({'error': 'The response took too long to generate. Please try again.'}), 504
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/health')
def health_check():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    if session_id in sessions:
        sessions[session_id]["history"] = []
        sessions[session_id]["system_prompt"] = (
            "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, "
            "and always strive to provide accurate information."
        )
        logger.info(f"Chat history has been reset for session {session_id[:8]}...")

    return jsonify({'status': 'success', 'message': 'Chat history has been reset.'})

if __name__ == "__main__":
    logger.info("Starting the Flask application")
    app.run(host='0.0.0.0', port=5000)