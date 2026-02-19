"""
AirCELA — Ollama Model Inference Example
==========================================

Load any Ollama-downloaded GGUF model for layer-by-layer inference.

Usage:
    python examples/ollama_inference.py mistral-small:22b
    python examples/ollama_inference.py mistral-small:22b mistralai/Mistral-Small-Instruct-2409
    python examples/ollama_inference.py /path/to/model.gguf

Developed by Gaurav Batule | https://github.com/gauravbatule/AirCELA
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add project root to path if running from examples dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def find_ollama_model(model_name: str):
    """
    Locate an Ollama model blob by name (e.g. 'mistral-small:22b').
    Returns the path to the GGUF blob file, or None.
    """
    home = Path.home()

    # Handle Windows / Linux / macOS paths for Ollama
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

    name, tag = name_tag.split(":", 1) if ":" in name_tag else (name_tag, "latest")

    manifest_path = (
        models_dir / "manifests" / "registry.ollama.ai" / library / name / tag
    )

    if not manifest_path.exists():
        print(f"  Error: Manifest not found at {manifest_path}")
        print(f"  Make sure you have pulled the model: ollama pull {model_name}")
        return None

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for layer in manifest.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                digest = layer.get("digest", "")
                if digest:
                    blob_hash = digest.replace("sha256:", "")
                    blob_path = models_dir / "blobs" / f"sha256-{blob_hash}"
                    if blob_path.exists():
                        return blob_path
                    else:
                        print(f"  Error: Blob file not found: {blob_path}")
                        return None
    except Exception as e:
        print(f"  Error parsing manifest: {e}")

    return None


def list_ollama_models():
    """List all available Ollama models."""
    home = Path.home()
    if sys.platform == "win32":
        manifests_dir = home / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    else:
        manifests_dir = Path("/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library")
        if not manifests_dir.exists():
            manifests_dir = home / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"

    models = []
    if manifests_dir.exists():
        for model_dir in sorted(manifests_dir.iterdir()):
            if model_dir.is_dir():
                for tag in sorted(model_dir.iterdir()):
                    try:
                        m = json.load(open(tag))
                        for layer in m.get("layers", []):
                            if "model" in layer.get("mediaType", ""):
                                d = layer["digest"].replace("sha256:", "")
                                bp = home / ".ollama" / "models" / "blobs" / f"sha256-{d}"
                                sz = bp.stat().st_size / 1e9 if bp.exists() else 0
                                models.append((model_dir.name, tag.name, sz))
                    except Exception:
                        pass
    return models


def main():
    if len(sys.argv) < 2:
        print("\n  AirCELA — Ollama Model Runner")
        print("  " + "=" * 40)
        print("\n  Usage:")
        print("    python examples/ollama_inference.py <model_name_or_path> [tokenizer_id]")
        print("\n  Examples:")
        print("    python examples/ollama_inference.py mistral-small:22b")
        print("    python examples/ollama_inference.py mistral-small:22b mistralai/Mistral-Small-Instruct-2409")
        print("    python examples/ollama_inference.py /path/to/model.gguf")

        # List available models
        models = list_ollama_models()
        if models:
            print("\n  Available Ollama models:")
            for name, tag, sz in models:
                print(f"    {name}:{tag}  ({sz:.1f} GB)")
        else:
            print("\n  No Ollama models found. Pull one first:")
            print("    ollama pull mistral-small:22b")
        return

    target = sys.argv[1]
    tokenizer_id = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if target is a direct file path
    if Path(target).exists() and Path(target).is_file():
        gguf_path = Path(target)
    else:
        # Try to find in Ollama
        print(f"\n  Searching for Ollama model: {target}...")
        gguf_path = find_ollama_model(target)

    if not gguf_path or not gguf_path.exists():
        print(f"  Error: Could not find model file for '{target}'")
        return

    print(f"  Loading model from: {gguf_path}")
    if tokenizer_id:
        print(f"  Using tokenizer: {tokenizer_id}")
    else:
        print("  Note: No tokenizer provided. Output will be raw token IDs.")
        print("  Tip: Add a tokenizer ID for readable text output:")
        print(f"    python examples/ollama_inference.py {target} <hf_tokenizer_id>")

    try:
        from aircela import CELAEngine

        engine = CELAEngine.from_gguf(str(gguf_path), tokenizer_id=tokenizer_id)

        if tokenizer_id:
            prompt = "Explain why layer-wise inference enables running large models on consumer GPUs."
            print(f"\n  Prompt: {prompt}\n")
            print("  Response: ", end="", flush=True)

            for token in engine.generate(prompt, max_tokens=100):
                print(token, end="", flush=True)
        else:
            # No tokenizer — use raw input_ids
            import torch
            input_ids = torch.tensor([[1, 4071, 28747]])  # BOS + sample tokens
            print(f"\n  Input IDs: {input_ids.tolist()}")
            print("  Generated IDs: ", end="", flush=True)

            for tok in engine.generate(input_ids=input_ids, max_tokens=20):
                print(tok + " ", end="", flush=True)

        print("\n\n  --- Generation Complete ---")
        print(f"  Support AirCELA: https://buymeacoffee.com/gauravbatule")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
