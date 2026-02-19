"""
CELA Basic Generation Example
==============================

Generate text from any HuggingFace model using CELA's
layer-by-layer inference engine.

Usage:
    python examples/basic_generation.py

Developed by Gaurav Batule | https://buymeacoffee.com/gauravbatule
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aircela import CELAEngine

def main():
    # Choose your model — ANY size works!
    # Small (fast):  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Medium:        "mistralai/Mistral-7B-v0.1"
    # Large:         "meta-llama/Llama-2-13b-hf"
    # Huge:          "meta-llama/Llama-2-70b-hf"
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = "What is the capital of France? Explain briefly."
    
    print(f"\n  Loading model: {model_name}")
    engine = CELAEngine.from_pretrained(model_name)
    
    print(f"\n  Generating response for: \"{prompt}\"\n")
    print("  " + "-" * 50)
    print("  ", end="")
    
    for token in engine.generate(prompt, max_tokens=80, temperature=0.7):
        print(token, end="", flush=True)
    
    print("\n  " + "-" * 50)
    print(f"\n  Done! Support CELA: https://buymeacoffee.com/gauravbatule")

if __name__ == "__main__":
    main()
