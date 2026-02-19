import sys
import os
from pathlib import Path
import traceback

# Add current dir to path
sys.path.insert(0, os.getcwd())

from aircela import CELAEngine

def find_ollama_model(model_name: str):
    """
    Locate an Ollama model blob by name (e.g. 'mistral-small:22b').
    Returns the path to the GGUF blob file.
    """
    import json
    
    home = Path.home()
    # Handle Windows/Linux paths for Ollama
    if sys.platform == "win32":
        models_dir = home / ".ollama" / "models"
    else:
        models_dir = Path("/usr/share/ollama/.ollama/models")
        if not models_dir.exists():
            models_dir = home / ".ollama" / "models"
            
    # Parse model name (library/model:tag)
    if ":" not in model_name:
        model_name += ":latest"
        
    parts = model_name.split("/")
    if len(parts) == 1:
        library = "library"
        name_tag = parts[0]
    else:
        library = parts[0]
        name_tag = parts[1]
        
    if ":" in name_tag:
        name, tag = name_tag.split(":")
    else:
        name, tag = name_tag, "latest"

    manifest_path = models_dir / "manifests" / "registry.ollama.ai" / library / name / tag
    
    if not manifest_path.exists():
        # Try without check? No, must exist.
        return None
        
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = layer.get("digest")
                if digest:
                    blob_hash = digest.replace("sha256:", "")
                    return models_dir / "blobs" / f"sha256-{blob_hash}"
    except Exception as e:
        print(f"Error parsing manifest: {e}")
        
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ollama.py <model_name_or_path> [tokenizer_id]")
        print("Example: python test_ollama.py mistral-small:22b mistralai/Mistral-Small-Instruct-2409")
        # specific hardcoded test for dev convenience if no args?
        # No, keep it clean for git.
        return

    target = sys.argv[1]
    tokenizer_id = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if target is a path
    if Path(target).exists() and Path(target).is_file():
        gguf_path = Path(target)
    else:
        # Try to find in Ollama
        print(f"Searching for Ollama model: {target}...")
        gguf_path = find_ollama_model(target)
        
    if not gguf_path or not gguf_path.exists():
        print(f"Error: Could not find model file for '{target}' or path does not exist.")
        return

    print(f"Loading model from: {gguf_path}")
    if tokenizer_id:
        print(f"Using tokenizer: {tokenizer_id}")
    else:
        print("Warning: No tokenizer provided. Output will be raw token IDs.")

    try:
        engine = CELAEngine.from_gguf(str(gguf_path), tokenizer_id=tokenizer_id)
        
        prompt = "Explain why layer-wise inference enables running large models on consumer GPUs."
        print(f"\nPrompt: {prompt}\n")
        print("Response: ", end="")
        
        for token in engine.generate(prompt, max_tokens=100):
            print(token, end="", flush=True)
            
        print("\n\n--- Generation Complete ---")
        
    except Exception as e:
        print(f"\n[!!!] Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
