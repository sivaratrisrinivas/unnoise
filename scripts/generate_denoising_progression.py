"""Capture every denoising step of a DDPM run on CPU."""

from pathlib import Path

import torch
from diffusers import DDPMPipeline
from diffusers.utils.torch_utils import randn_tensor

MODEL_ID = "google/ddpm-cifar10-32"
OUTPUT_ROOT = Path("outputs/diffusion_progression")
FRAMES_DIR = OUTPUT_ROOT / "frames"
FINAL_IMAGE_PATH = OUTPUT_ROOT / "final_image.png"
NUM_INFERENCE_STEPS = 50
SEED = 7


def load_pipeline() -> DDPMPipeline:
    print(f"Loading {MODEL_ID} on CPU...")
    pipe = DDPMPipeline.from_pretrained(MODEL_ID)
    return pipe.to("cpu")


def tensor_to_pil(pipe: DDPMPipeline, image: torch.Tensor):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return pipe.numpy_to_pil(image)[0]


def prepare_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for existing_frame in FRAMES_DIR.glob("frame_*.png"):
        existing_frame.unlink()


def capture_progression(pipe: DDPMPipeline, num_inference_steps: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if isinstance(pipe.unet.config.sample_size, int):
        image_shape = (
            1,
            pipe.unet.config.in_channels,
            pipe.unet.config.sample_size,
            pipe.unet.config.sample_size,
        )
    else:
        image_shape = (1, pipe.unet.config.in_channels, *pipe.unet.config.sample_size)

    if pipe.device.type == "mps":
        image = randn_tensor(image_shape, generator=generator, dtype=pipe.unet.dtype)
        image = image.to(pipe.device)
    else:
        image = randn_tensor(
            image_shape,
            generator=generator,
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )

    frames = [tensor_to_pil(pipe, image)]

    pipe.scheduler.set_timesteps(num_inference_steps)
    with torch.inference_mode():
        for timestep in pipe.progress_bar(pipe.scheduler.timesteps):
            model_output = pipe.unet(image, timestep).sample
            image = pipe.scheduler.step(
                model_output,
                timestep,
                image,
                generator=generator,
            ).prev_sample
            frames.append(tensor_to_pil(pipe, image))

    return frames


def save_frames(frames) -> None:
    for frame_index, frame in enumerate(frames):
        frame.save(FRAMES_DIR / f"frame_{frame_index:03d}.png")


def main() -> None:
    prepare_output_dirs()

    pipe = load_pipeline()
    print(f"Capturing {NUM_INFERENCE_STEPS} denoising steps...")
    frames = capture_progression(pipe, NUM_INFERENCE_STEPS, SEED)

    save_frames(frames)
    frames[-1].save(FINAL_IMAGE_PATH)

    print(f"Saved {len(frames)} frames to {FRAMES_DIR.resolve()}")
    print(f"Saved final image to {FINAL_IMAGE_PATH.resolve()}")


if __name__ == "__main__":
    main()
