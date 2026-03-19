"""Generate one image from a prompt with a CPU-only text-to-image pipeline."""

import argparse
from pathlib import Path

from diffusion_core import DEFAULT_SEED, DEFAULT_STEPS, generate_final_image

DEFAULT_PROMPT = "a red bicycle on a rainy street"
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "generate_single_image.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating a prompt-driven image for: {args.prompt!r}")
    image = generate_final_image(args.prompt, args.steps, args.seed)

    image.save(OUTPUT_PATH)
    print(f"Saved image to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
