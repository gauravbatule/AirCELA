"""
AirCELA CLI — Command-line interface for AirLLM-style optimized inference.

Usage:
    aircela chat -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    aircela run -m "mistralai/Mistral-7B-v0.1" -p "Hello!"
    aircela info
    aircela bench

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import sys
import os
import argparse
import time

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"


def cmd_info(args):
    """Show system information and hardware capabilities."""
    from aircela import __version__
    from aircela.device import DeviceManager

    dm = DeviceManager()
    print()
    print("=" * 60)
    print(f"  AirCELA v{__version__} — Fastest AirLLM Alternative")
    print(f"  Developed by Gaurav Batule")
    print("=" * 60)
    print(f"  {dm.summary()}")
    print()


def cmd_chat(args):
    """Interactive chat with a HuggingFace model."""
    from aircela.engine import CELAEngine

    print()
    print("=" * 60)
    print("  CELA Interactive Chat")
    print("=" * 60)

    engine = CELAEngine.from_pretrained(args.model)

    print(f"\n  Type your message. Commands: /quit, /clear, /reset")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            break
        if user_input.lower() in ("/clear", "/reset"):
            engine.reset()
            print("  Context cleared.")
            continue

        sys.stdout.write("  AI: ")
        sys.stdout.flush()
        for token in engine.generate(
            user_input,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ):
            sys.stdout.write(token)
            sys.stdout.flush()

    print("\n  Goodbye!")


def cmd_run(args):
    """Single prompt generation."""
    from aircela.engine import CELAEngine

    engine = CELAEngine.from_pretrained(args.model)

    for token in engine.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()


def cmd_bench(args):
    """Benchmark hardware and model performance."""
    from aircela import __version__
    from aircela.device import DeviceManager

    dm = DeviceManager()

    print()
    print("=" * 60)
    print(f"  AirCELA v{__version__} Performance Benchmark")
    print("=" * 60)
    print(f"  {dm.summary()}")

    # RAM bandwidth
    import numpy as np

    print(f"\n  [1/3] RAM Bandwidth")
    for size_mb in [64, 256]:
        data = np.random.randn(size_mb * 1024 * 1024 // 8).astype(np.float64)
        t0 = time.perf_counter()
        _ = data.copy()
        dt = time.perf_counter() - t0
        bw = size_mb / dt / 1024
        print(f"    {size_mb:4d} MB: {dt*1000:.1f}ms ({bw:.1f} GB/s)")

    # GPU bandwidth
    if dm.has_cuda:
        import torch

        print(f"\n  [2/3] GPU Performance")
        x = torch.randn(64, 1024, 1024, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y = x.clone()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        bw = x.nelement() * 2 / dt / 1e9
        print(f"    GPU bandwidth: {bw:.0f} GB/s")

        # GEMM
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        gflops = 2 * 4096 ** 3 / dt / 1e9
        print(f"    GEMM 4096x4096: {dt*1000:.1f}ms ({gflops:.0f} GFLOPS)")

        del x, y, a, b, c
        torch.cuda.empty_cache()

    # SSD speed
    print(f"\n  [3/3] SSD Read Speed")
    test_size = 100 * 1024 * 1024
    test_data = os.urandom(test_size)
    test_file = "__cela_bench_tmp__"
    with open(test_file, "wb") as f:
        f.write(test_data)
    t0 = time.perf_counter()
    with open(test_file, "rb") as f:
        _ = f.read()
    dt = time.perf_counter() - t0
    bw = test_size / dt / 1e6
    os.remove(test_file)
    print(f"    100 MB read: {dt*1000:.0f}ms ({bw:.0f} MB/s)")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        prog="aircela",
        description="AirCELA — Fastest AirLLM-style inference pipeline | By Gaurav Batule",
    )
    sub = parser.add_subparsers(dest="command")

    # info
    sub.add_parser("info", help="Show hardware info")

    # chat
    chat_p = sub.add_parser("chat", help="Interactive chat")
    chat_p.add_argument("-m", "--model", required=True, help="HuggingFace model ID")
    chat_p.add_argument("-n", "--max-tokens", type=int, default=200)
    chat_p.add_argument("-t", "--temperature", type=float, default=0.7)

    # run
    run_p = sub.add_parser("run", help="Single prompt")
    run_p.add_argument("-m", "--model", required=True, help="HuggingFace model ID")
    run_p.add_argument("-p", "--prompt", required=True, help="Input prompt")
    run_p.add_argument("-n", "--max-tokens", type=int, default=100)
    run_p.add_argument("-t", "--temperature", type=float, default=0.7)

    # bench
    sub.add_parser("bench", help="Benchmark hardware")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "bench":
        cmd_bench(args)
    else:
        parser.print_help()
        print("\n  Quick start:")
        print('    cela chat -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0"')
        print("    cela info")
        print("    cela bench")


if __name__ == "__main__":
    main()
