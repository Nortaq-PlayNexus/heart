"""H.E.A.R.T. command-line interface.

Usage:
    python -m heart.cli "I'm feeling really happy today!"
    python -m heart.cli --profile
    python -m heart.cli --interactive
    python -m heart.cli --batch file.txt
"""

import argparse
import json
import sys

from heart.core.engine import HeartEngine
from heart.core.config import HeartConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="heart",
        description="H.E.A.R.T. - Human-like Emotional Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m heart.cli "I feel great today"\n'
            "  python -m heart.cli --profile\n"
            "  python -m heart.cli --interactive\n"
        ),
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to analyze for emotional content",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show the current emotional profile",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive session",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Process a file of texts (one per line)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=1000,
        help="Working memory capacity (default: 1000)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=10,
        help="Context window size (default: 10)",
    )
    return parser


def process_text(engine: HeartEngine, text: str, use_json: bool) -> None:
    """Process a single text input and display results."""
    result = engine.respond(text)

    if use_json:
        print(json.dumps(result, indent=2))
    else:
        state = result["emotional_state"]
        print(f"\n--- Emotional Analysis ---")
        print(f"Text:          {text}")
        print(f"Primary:       {result['primary_emotion']}")
        print(f"Intensity:     {result['intensity']:.2f}")
        print(f"Sentiment:     {result['sentiment']}")
        print(f"Valence:       {state['valence']:.3f}")
        print(f"Arousal:       {state['arousal']:.3f}")
        print(f"Dominance:     {state['dominance']:.3f}")
        print(f"Blended:       {'Yes' if result['is_blended'] else 'No'}")
        print(f"\nResponse:      {result['response_text']}")
        print()


def show_profile(engine: HeartEngine) -> None:
    """Display the current emotional profile."""
    profile = engine.get_emotional_profile()
    print("\n--- Emotional Profile ---")
    for key, value in profile.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.3f}")
        elif isinstance(value, list):
            print(f"  {key:25s}: {', '.join(str(v) for v in value) if value else '(none)'}")
        else:
            print(f"  {key:25s}: {value}")
    print()


def interactive_mode(engine: HeartEngine, use_json: bool) -> None:
    """Run an interactive session."""
    print("H.E.A.R.T. Interactive Mode")
    print("Type 'quit' or 'exit' to stop, 'profile' to see emotional profile.\n")

    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if text.lower() == "profile":
            show_profile(engine)
            continue

        process_text(engine, text, use_json)


def batch_mode(engine: HeartEngine, filepath: str, use_json: bool) -> None:
    """Process a file of texts."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    for i, line in enumerate(lines, 1):
        print(f"\n--- Line {i} ---")
        process_text(engine, line, use_json)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    config = HeartConfig(
        working_memory_size=args.memory_size,
        context_window=args.context_window,
    )
    engine = HeartEngine(config=config)

    if args.profile:
        show_profile(engine)
    elif args.interactive:
        interactive_mode(engine, args.json)
    elif args.batch:
        batch_mode(engine, args.batch, args.json)
    elif args.text:
        process_text(engine, args.text, args.json)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
