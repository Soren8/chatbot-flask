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
llm = Llama(model_path=model_path, verbose=False)

# Define system prompt and user prompt
system_prompt = "You are a helpful AI assistant named Claude. You are knowledgeable, friendly, and always strive to provide accurate information."
user_prompt = "Tell me about the importance of renewable energy."

# Combine system prompt and user prompt
full_prompt = f"""### System:
{system_prompt}

### Human: {user_prompt}

### Assistant: """

# Generate text
output = llm(full_prompt, max_tokens=10000, echo=False, stop=["### Human:", "\n\n"])

# Print the generated text
print(output['choices'][0]['text'])