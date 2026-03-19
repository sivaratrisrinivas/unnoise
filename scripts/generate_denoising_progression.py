"""Capture every denoising step of a prompt-driven diffusion run on CPU."""

import argparse
from pathlib import Path

from diffusion_core import DEFAULT_SEED, DEFAULT_STEPS, capture_progression

DEFAULT_PROMPT = "a red bicycle on a rainy street"
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dirs()

    print(f"Capturing {args.steps} denoising steps for: {args.prompt!r}")
    frames = capture_progression(args.prompt, args.steps, args.seed)

    save_frames(frames)
    frames[-1].save(FINAL_IMAGE_PATH)

    print(f"Saved {len(frames)} frames to {FRAMES_DIR.resolve()}")
    print(f"Saved final image to {FINAL_IMAGE_PATH.resolve()}")


if __name__ == "__main__":
    main()
