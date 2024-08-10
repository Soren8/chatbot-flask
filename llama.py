from llama_cpp import Llama

# Initialize the model
# Replace 'path/to/your/model.bin' with the actual path to your downloaded model
llm = Llama(model_path="F:\lmstudio\cognitivecomputations\dolphin-2.9-llama3-8b-gguf\dolphin-2.9-llama3-8b-q6_K.gguf")

# Generate text
prompt = "Once upon a time, in a land far away,"
output = llm(prompt, max_tokens=10000)

# Print the generated text
print(output['choices'][0]['text'])