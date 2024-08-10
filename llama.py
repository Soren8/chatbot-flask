import os
import sys
import logging
from llama_cpp import Llama

# Redirect C++ stdout/stderr to /dev/null
os.environ['LLAMA_CPP_VERBOSE'] = '0'

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Suppress warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Initialize the model with the provided path
model_path = r"F:\lmstudio\cognitivecomputations\dolphin-2.9-llama3-8b-gguf\dolphin-2.9-llama3-8b-q6_K.gguf"
llm = Llama(model_path=model_path, verbose=False, n_ctx=8192)

def get_system_prompt():
    default_prompt = "You are a helpful AI assistant named Llama. You are knowledgeable, friendly, and always strive to provide accurate information."
    print("\nEnter a custom system prompt, or press Enter to use the default:")
    print(f"Default: {default_prompt}")
    user_input = input("Custom prompt: ").strip()
    return user_input if user_input else default_prompt

def generate_response(prompt, conversation_history, system_prompt):
    full_prompt = f"""{system_prompt}

{conversation_history}

### Human: {prompt}

### Assistant: """
    
    output = llm(full_prompt, max_tokens=10000, echo=False, stop=["### Human:", "\n\n"])
    return output['choices'][0]['text'].strip()

def main():
    system_prompt = get_system_prompt()
    print("\nWelcome to the AI chat! Type 'exit' to end the conversation.")
    conversation_history = ""
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Thank you for chatting. Goodbye!")
            break
        
        response = generate_response(user_input, conversation_history, system_prompt)
        print(f"\nAI: {response}")
        
        # Update conversation history
        conversation_history += f"\n### Human: {user_input}\n\n### Assistant: {response}\n"

if __name__ == "__main__":
    main()