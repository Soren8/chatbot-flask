import llama_cpp
import torch

# Check if torch can detect CUDA
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# Attempt to initialize llama_cpp with GPU layers
try:
    from llama_cpp import Llama
    llm = Llama(
        model_path=r"F:\lmstudio\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        n_ctx=16384,
        n_gpu_layers=10,  # Set to a positive number
        verbose=True  # Enable verbose to see detailed logs
    )
    print("llama_cpp initialized successfully with GPU support.")
except Exception as e:
    print(f"Error initializing llama_cpp with GPU support: {e}")
