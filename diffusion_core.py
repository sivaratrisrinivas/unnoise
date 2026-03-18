"""Shared diffusion utilities for the tutorial app and scripts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from diffusers import DDPMPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "google/ddpm-cifar10-32"


@dataclass(frozen=True)
class SavedProgression:
    frame_paths: list[Path]
    final_image_path: Path


@lru_cache(maxsize=1)
def load_pipeline() -> DDPMPipeline:
    """Load the CPU-only DDPM pipeline once per process."""
    pipe = DDPMPipeline.from_pretrained(MODEL_ID)
    pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _image_shape(pipe: DDPMPipeline, batch_size: int = 1) -> tuple[int, ...]:
    if isinstance(pipe.unet.config.sample_size, int):
        return (
            batch_size,
            pipe.unet.config.in_channels,
            pipe.unet.config.sample_size,
            pipe.unet.config.sample_size,
        )

    return (batch_size, pipe.unet.config.in_channels, *pipe.unet.config.sample_size)


def tensor_to_pil(pipe: DDPMPipeline, image: torch.Tensor) -> Image.Image:
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return pipe.numpy_to_pil(image)[0]


def frame_label(index: int) -> str:
    return "noise" if index == 0 else f"step {index}"


def capture_progression(
    num_inference_steps: int = 50,
    seed: int = 7,
    *,
    show_progress_bar: bool = False,
) -> list[Image.Image]:
    """Run the DDPM loop and keep the initial noise plus each intermediate frame."""
    pipe = load_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if pipe.device.type == "mps":
        image = randn_tensor(_image_shape(pipe), generator=generator, dtype=pipe.unet.dtype)
        image = image.to(pipe.device)
    else:
        image = randn_tensor(
            _image_shape(pipe),
            generator=generator,
            device=pipe.device,
            dtype=pipe.unet.dtype,
        )

    frames = [tensor_to_pil(pipe, image)]
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    iterator = pipe.progress_bar(timesteps) if show_progress_bar else timesteps

    with torch.inference_mode():
        for timestep in iterator:
            model_output = pipe.unet(image, timestep).sample
            image = pipe.scheduler.step(
                model_output,
                timestep,
                image,
                generator=generator,
            ).prev_sample
            frames.append(tensor_to_pil(pipe, image))

    return frames


def build_contact_sheet(
    frames: list[Image.Image],
    *,
    columns: int = 6,
    thumb_size: int = 96,
    padding: int = 12,
    caption_height: int = 16,
) -> Image.Image:
    """Build a simple contact sheet for the progression frames."""
    from math import ceil

    rows = ceil(len(frames) / columns)
    cell_width = thumb_size + padding * 2
    cell_height = thumb_size + caption_height + padding * 2
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        x = column * cell_width + padding
        y = row * cell_height + padding
        thumb = frame.resize((thumb_size, thumb_size), Image.Resampling.NEAREST)
        canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb_size + 2), frame_label(index), fill="black", font=font)

    return canvas


def save_progression_frames(frames: list[Image.Image], run_dir: Path) -> SavedProgression:
    """Save the frame sequence and the final image for a progression run."""
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for index, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{index:03d}.png"
        frame.save(frame_path)
        frame_paths.append(frame_path)

    final_image_path = run_dir / "final_image.png"
    frames[-1].save(final_image_path)

    return SavedProgression(
        frame_paths=frame_paths,
        final_image_path=final_image_path,
    )

