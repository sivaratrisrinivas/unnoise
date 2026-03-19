"""Generate one image from either a direct prompt or a thought-completion fragment."""

import argparse
from pathlib import Path

from diffusion_core import (
    DEFAULT_SEED,
    DEFAULT_STEPS,
    PROMPT_MODE_DIRECT,
    SUPPORTED_PROMPT_MODES,
    condition_prompt,
    generate_final_image,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "generate_single_image.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", choices=SUPPORTED_PROMPT_MODES, default=PROMPT_MODE_DIRECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditioned = condition_prompt(args.prompt, mode=args.mode)

    print(f"Generating {conditioned.mode_label.lower()} image for: {conditioned.seed_text!r}")
    if conditioned.mode != PROMPT_MODE_DIRECT:
        print(f"Conditioned prompt: {conditioned.resolved_prompt}")
    image = generate_final_image(conditioned.resolved_prompt, args.steps, args.seed)

    image.save(OUTPUT_PATH)
    print(f"Saved image to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
