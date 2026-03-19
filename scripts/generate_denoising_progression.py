"""Capture every denoising step of a conditioned diffusion run on CPU."""

import argparse
from pathlib import Path

from diffusion_core import (
    DEFAULT_SEED,
    DEFAULT_STEPS,
    PROMPT_MODE_DIRECT,
    SUPPORTED_PROMPT_MODES,
    capture_progression,
    condition_prompt,
)

OUTPUT_ROOT = Path("outputs/diffusion_progression")
FRAMES_DIR = OUTPUT_ROOT / "frames"
FINAL_IMAGE_PATH = OUTPUT_ROOT / "final_image.png"


def prepare_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for existing_frame in FRAMES_DIR.glob("frame_*.png"):
        existing_frame.unlink()


def save_frames(frames) -> None:
    for frame_index, frame in enumerate(frames):
        frame.save(FRAMES_DIR / f"frame_{frame_index:03d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", choices=SUPPORTED_PROMPT_MODES, default=PROMPT_MODE_DIRECT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dirs()
    conditioned = condition_prompt(args.prompt, mode=args.mode)

    print(f"Capturing {args.steps} denoising steps for: {conditioned.seed_text!r}")
    if conditioned.mode != PROMPT_MODE_DIRECT:
        print(f"Conditioned prompt: {conditioned.resolved_prompt}")
    frames = capture_progression(conditioned.resolved_prompt, args.steps, args.seed)

    save_frames(frames)
    frames[-1].save(FINAL_IMAGE_PATH)

    print(f"Saved {len(frames)} frames to {FRAMES_DIR.resolve()}")
    print(f"Saved final image to {FINAL_IMAGE_PATH.resolve()}")


if __name__ == "__main__":
    main()
