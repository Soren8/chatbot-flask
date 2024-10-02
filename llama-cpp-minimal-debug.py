import sys
import torch
from llama_cpp import Llama
import GPUtil
import os
import llama_cpp

def check_pytorch_cuda():
    print("=== PyTorch CUDA Check ===")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available in PyTorch.")
    print("===========================\n")

def check_llama_cpp_cuda(model_path, n_gpu_layers):
    print("=== Initializing LLaMA with CUDA Support ===")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=512,  # Small context for minimal example
            n_gpu_layers=10,
            verbose=True  # Enable verbose logging for detailed output
        )
        print("LLaMA initialized successfully with GPU support.\n")
        return llm
    except Exception as e:
        print(f"Error initializing LLaMA with CUDA support: {e}")
        sys.exit(1)

def monitor_gpu_usage(stage="Before Model Initialization"):
    print(f"=== GPU Usage {stage} ===")
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Load: {gpu.load*100:.2f}%")
        print(f"  Free Memory: {gpu.memoryFree}MB")
        print(f"  Used Memory: {gpu.memoryUsed}MB")
        print(f"  Total Memory: {gpu.memoryTotal}MB\n")
    print("===========================\n")

def generate_simple_response(llm):
    print("=== Generating Simple Response ===")
    prompt = "Hello, how are you?"
    try:
        response = llm(prompt, max_tokens=50, stop=["\n"], stream=False)
        print("Response:", response['choices'][0]['text'].strip())
    except Exception as e:
        print(f"Error during response generation: {e}")
    print("===========================\n")

def main():
    print(f"LLAMA_CPP_CUBLAS: {os.getenv('LLAMA_CPP_CUBLAS')}")
    print(f"llama_cpp.__version__: {llama_cpp.__version__}")

    # Path to your model
    model_path = r"F:\lmstudio\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
    
    # Determine number of GPU layers to offload
    n_gpu_layers = 10  # Start by offloading 10 layers to GPU

    # Step 1: Check GPU usage before initialization
    print("Monitoring GPU usage before model initialization:")
    monitor_gpu_usage("Before Model Initialization")
    
    # Step 2: Check PyTorch CUDA availability
    check_pytorch_cuda()
    
    # Step 3: Initialize LLaMA with CUDA support
    llm = check_llama_cpp_cuda(model_path, n_gpu_layers)
    
    # Step 4: Monitor GPU usage after initialization
    print("Monitoring GPU usage after model initialization:")
    monitor_gpu_usage("After Model Initialization")
    
    # Step 5: Generate a simple response to trigger GPU usage
    generate_simple_response(llm)
    
    # Step 6: Monitor GPU usage after response generation
    print("Monitoring GPU usage after response generation:")
    monitor_gpu_usage("After Response Generation")

if __name__ == "__main__":
    main()
