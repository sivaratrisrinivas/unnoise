"""Generate one image from pure noise with a CPU-only DDPM pipeline."""

from pathlib import Path

import torch
from diffusers import DDPMPipeline

MODEL_ID = "google/ddpm-cifar10-32"
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "step2_single_image.png"
NUM_INFERENCE_STEPS = 50
SEED = 7


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} on CPU...")
    pipe = DDPMPipeline.from_pretrained(MODEL_ID)
    pipe = pipe.to("cpu")

    print(
        f"Generating one image from pure noise with {NUM_INFERENCE_STEPS} denoising steps..."
    )
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    image = pipe(
        generator=generator,
        num_inference_steps=NUM_INFERENCE_STEPS,
    ).images[0]

    image.save(OUTPUT_PATH)
    print(f"Saved image to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
