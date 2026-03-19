"""Shared text-to-image diffusion utilities for the tutorial app and scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from transformers import logging as transformers_logging

MODEL_ID = "OFA-Sys/small-stable-diffusion-v0"
DEFAULT_STEPS = 10
MIN_STEPS = 5
MAX_STEPS = 20
DEFAULT_SEED = 7
DEFAULT_GUIDANCE_SCALE = 7.5
HF_TOKEN_FILE = Path(__file__).resolve().parent / ".env.local"


@dataclass(frozen=True)
class SavedProgression:
    frame_paths: list[Path]
    final_image_path: Path


def _read_hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token.strip()

    if not HF_TOKEN_FILE.is_file():
        return None

    for raw_line in HF_TOKEN_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key, separator, value = line.partition("=")
        if separator and key.strip() in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            cleaned = value.strip().strip('"').strip("'")
            if cleaned:
                return cleaned

    return None


def _build_scheduler(local_files_only: bool, token: str | None) -> DPMSolverMultistepScheduler:
    scheduler_config_path = hf_hub_download(
        MODEL_ID,
        filename="scheduler/scheduler_config.json",
        local_files_only=local_files_only,
        token=token,
    )
    config = json.loads(Path(scheduler_config_path).read_text())
    config.pop("predict_epsilon", None)
    config["steps_offset"] = 1
    config["timestep_spacing"] = "trailing"
    return DPMSolverMultistepScheduler.from_config(config)


@lru_cache(maxsize=1)
def load_pipeline() -> StableDiffusionPipeline:
    """Load the CPU-only text-to-image pipeline once per process."""

    hf_token = _read_hf_token()
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    transformers_verbosity = transformers_logging.get_verbosity()
    diffusers_verbosity = diffusers_logging.get_verbosity()
    last_error: Exception | None = None
    transformers_logging.set_verbosity_error()
    diffusers_logging.set_verbosity_error()
    try:
        for local_files_only in (True, False):
            try:
                scheduler = _build_scheduler(local_files_only, hf_token)
                pipe = StableDiffusionPipeline.from_pretrained(
                    MODEL_ID,
                    scheduler=scheduler,
                    local_files_only=local_files_only,
                    low_cpu_mem_usage=False,
                    use_safetensors=False,
                    token=hf_token,
                )
                pipe.enable_attention_slicing()
                pipe.vae.enable_slicing()
                pipe.set_progress_bar_config(disable=True)
                return pipe.to("cpu")
            except (FileNotFoundError, OSError, ValueError) as exc:
                last_error = exc
                continue
    finally:
        transformers_logging.set_verbosity(transformers_verbosity)
        diffusers_logging.set_verbosity(diffusers_verbosity)

    assert last_error is not None
    raise last_error


def _latent_shape(pipe: StableDiffusionPipeline, batch_size: int = 1) -> tuple[int, ...]:
    if isinstance(pipe.unet.config.sample_size, int):
        return (
            batch_size,
            pipe.unet.config.in_channels,
            pipe.unet.config.sample_size,
            pipe.unet.config.sample_size,
        )

    return (batch_size, pipe.unet.config.in_channels, *pipe.unet.config.sample_size)


def tensor_to_pil(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
    with torch.inference_mode():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def frame_label(index: int) -> str:
    return "noise" if index == 0 else f"step {index}"


def capture_progression(
    prompt: str,
    num_inference_steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
) -> list[Image.Image]:
    """Run the text-to-image diffusion loop and keep the initial noise plus each frame."""
    pipe = load_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    latents = randn_tensor(
        _latent_shape(pipe),
        generator=generator,
        device=pipe.device,
        dtype=pipe.unet.dtype,
    )

    latent_history = [latents.detach().cpu().clone()]

    def capture_callback(_pipe, _step_index, _timestep, callback_kwargs):
        latent_history.append(callback_kwargs["latents"].detach().cpu().clone())
        return callback_kwargs

    with torch.inference_mode():
        pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            generator=generator,
            latents=latents,
            callback_on_step_end=capture_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

    frames = [tensor_to_pil(pipe, latents) for latents in latent_history]

    return frames


def generate_final_image(
    prompt: str,
    num_inference_steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
) -> Image.Image:
    """Generate just the final image for a single prompt."""
    return capture_progression(prompt, num_inference_steps, seed)[-1]


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
        thumb = frame.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
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
