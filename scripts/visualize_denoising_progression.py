"""Build a contact sheet from saved diffusion progression frames."""

from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from diffusion_core import frame_label

FRAMES_DIR = Path("outputs/diffusion_progression/frames")
CONTACT_SHEET_PATH = Path("outputs/diffusion_progression/contact_sheet.png")
COLUMNS = 6
THUMB_SIZE = 96
PADDING = 12
CAPTION_HEIGHT = 16


def load_frames() -> list[Image.Image]:
    frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No saved frames found in {FRAMES_DIR}")

    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    return frames


def build_contact_sheet(frames: list[Image.Image]) -> Image.Image:
    rows = ceil(len(frames) / COLUMNS)
    cell_width = THUMB_SIZE + PADDING * 2
    cell_height = THUMB_SIZE + CAPTION_HEIGHT + PADDING * 2
    canvas = Image.new("RGB", (COLUMNS * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, frame in enumerate(frames):
        row, column = divmod(index, COLUMNS)
        x = column * cell_width + PADDING
        y = row * cell_height + PADDING
        thumb = frame.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.NEAREST)
        canvas.paste(thumb, (x, y))
        draw.text((x, y + THUMB_SIZE + 2), frame_label(index), fill="black", font=font)

    return canvas


def main() -> None:
    frames = load_frames()
    contact_sheet = build_contact_sheet(frames)
    CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(CONTACT_SHEET_PATH)
    print(f"Saved contact sheet to {CONTACT_SHEET_PATH.resolve()}")


if __name__ == "__main__":
    main()
