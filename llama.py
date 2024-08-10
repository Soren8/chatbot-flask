import os
import sys
import logging
from llama_cpp import Llama
from flask import Flask, render_template, request, jsonify
from functools import wraps
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define context length
CONTEXT_LENGTH = 512

# Initialize the model with the provided path and increased context length
model_path = r"F:\lmstudio\cognitivecomputations\dolphin-2.9-llama3-8b-gguf\dolphin-2.9-llama3-8b-q6_K.gguf"
llm = Llama(model_path=model_path, n_ctx=CONTEXT_LENGTH)

# Create Flask app
app = Flask(__name__)

default_system_prompt = "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, and always strive to provide accurate information."
conversation_history = []
current_system_prompt = default_system_prompt

# Approximate token counts (these are rough estimates, adjust as needed)
MAX_TOKENS = int(CONTEXT_LENGTH * 0.9) # Leave some room for the new prompt and response
TOKENS_PER_MESSAGE = 4  # Rough estimate for "### Human: " and "### Assistant: "
APPROXIMATE_TOKENS_PER_WORD = 1.3

def estimate_tokens(text):
    return int(len(text.split()) * APPROXIMATE_TOKENS_PER_WORD)

def manage_conversation_history(new_prompt, new_response):
    global conversation_history
    
    # Add new exchange to history
    conversation_history.append((new_prompt, new_response))
    
    # Calculate total tokens
    total_tokens = sum(estimate_tokens(prompt) + estimate_tokens(response) + TOKENS_PER_MESSAGE 
                       for prompt, response in conversation_history)
    
    # Remove oldest exchanges if we exceed the limit
    while total_tokens > MAX_TOKENS and conversation_history:
        removed_prompt, removed_response = conversation_history.pop(0)
        total_tokens -= (estimate_tokens(removed_prompt) + estimate_tokens(removed_response) + TOKENS_PER_MESSAGE)
        logger.info(f"Removed oldest exchange to maintain context length. Current estimated tokens: {total_tokens}")

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
def generate_response(prompt, system_prompt):
    global conversation_history
    
    history_text = "\n".join([f"### Human: {p}\n\n### Assistant: {r}" for p, r in conversation_history])
    
    full_prompt = f"""{system_prompt}

{history_text}

### Human: {prompt}

### Assistant: """
    
    try:
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        logger.info(f"Using system prompt: {system_prompt[:50]}...")
        output = llm(full_prompt, max_tokens=10000, echo=False, stop=["### Human:"])
        response = output['choices'][0]['text'].strip()
        
        # Update conversation history
        manage_conversation_history(prompt, response)
        
        logger.info(f"Response generated successfully. Length: {len(response)} characters")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

@app.route('/')
def home():
    logger.info("Serving home page")
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    global current_system_prompt
    user_message = request.json['message']
    new_system_prompt = request.json.get('system_prompt')
    
    logger.info(f"Received chat request. Message: {user_message[:50]}...")
    
    if new_system_prompt:
        logger.info(f"Updating system prompt to: {new_system_prompt[:50]}...")
        current_system_prompt = new_system_prompt
    
    try:
        ai_response = generate_response(user_message, current_system_prompt)
        logger.info(f"Chat response generated successfully. Length: {len(ai_response)} characters")
        logger.info("Response preview:")
        logger.info(ai_response[:200] + "..." if len(ai_response) > 200 else ai_response)
        return jsonify({'response': ai_response, 'system_prompt': current_system_prompt})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

@app.route('/health')
def health_check():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

if __name__ == "__main__":
    logger.info("Starting the Flask application")
    app.run(host='0.0.0.0', port=5000)