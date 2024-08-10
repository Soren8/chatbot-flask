import os
import sys
import logging
from llama_cpp import Llama
from flask import Flask, render_template, request, jsonify

# Redirect C++ stdout/stderr to /dev/null
os.environ['LLAMA_CPP_VERBOSE'] = '0'

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Suppress warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Initialize the model with the provided path and increased context length
model_path = r"F:\lmstudio\cognitivecomputations\dolphin-2.9-llama3-8b-gguf\dolphin-2.9-llama3-8b-q6_K.gguf"
llm = Llama(model_path=model_path, verbose=False, n_ctx=8192)

app = Flask(__name__)

default_system_prompt = "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, and always strive to provide accurate information."
conversation_history = ""
current_system_prompt = default_system_prompt

def generate_response(prompt):
    global current_system_prompt
    global conversation_history
    full_prompt = f"""{current_system_prompt}

{conversation_history}

### Human: {prompt}

### Assistant: """
    
    output = llm(full_prompt, max_tokens=10000, echo=False, stop=["### Human:", "\n\n"])
    response = output['choices'][0]['text'].strip()
    
    # Update conversation history
    conversation_history += f"\n### Human: {prompt}\n\n### Assistant: {response}\n"
    return response

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    global current_system_prompt
    user_message = request.json['message']
    new_system_prompt = request.json.get('system_prompt')
    
    if new_system_prompt:
        current_system_prompt = new_system_prompt
    
    ai_response = generate_response(user_message)
    return jsonify({'response': ai_response, 'system_prompt': current_system_prompt})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)