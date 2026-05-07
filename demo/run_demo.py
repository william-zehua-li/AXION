import argparse
import json
import os
import sys

import torch

# Ensure the project root is importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import Config
from core.pipeline import Pipeline


def load_pipeline(checkpoint_path: str, config_overrides: dict = None) -> Pipeline:
    config = Config()
    if config_overrides:
        for key, value in config_overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown Config key: '{key}'")
            setattr(config, key, value)

    pipeline = Pipeline(config)

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # FIX: Pipeline is not an nn.Module — use pipeline.load_checkpoint()
        #      instead of the non-existent pipeline.load_state_dict()
        pipeline.load_checkpoint(checkpoint_path)
    else:
        print("No checkpoint supplied — running with randomly initialised weights.")

    return pipeline


def run_interactive(pipeline: Pipeline) -> None:
    print("AXION — interactive demo")
    print("Type 'exit' or 'quit' to stop, Ctrl-C to abort.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        result = pipeline.run(query)

        print(f"\nAnswer:      {result['answer']}")
        print(f"Ticks used:  {result['ticks_used']}")

        ku = result.get("knowledge_updates", {})
        written = ku.get("written", [])
        queued  = ku.get("queued", [])
        if written:
            print(f"Knowledge written:  {len(written)} item(s)")
        if queued:
            print(f"Knowledge queued:   {len(queued)} item(s) (pending review)")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AXION interactive demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a saved checkpoint file (omit to run with random weights)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single-query mode: print the answer for this query and exit",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help='JSON string of Config field overrides, e.g. \'{"max_ticks": 4}\'',
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (overrides Config.temperature)",
    )
    parser.add_argument(
        "--max_ticks", type=int, default=None,
        help="Maximum loop iterations (overrides Config.max_ticks)",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.config:
        overrides.update(json.loads(args.config))
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_ticks is not None:
        overrides["max_ticks"] = args.max_ticks

    pipeline = load_pipeline(args.checkpoint, overrides or None)

    if args.query:
        result = pipeline.run(args.query)
        print(result["answer"])
    else:
        run_interactive(pipeline)
