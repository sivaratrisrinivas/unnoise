# unnoise

`unnoise` is a small CPU-only diffusion project that shows how random noise slowly turns into a real image.

The whole point of the project is to make diffusion feel understandable instead of magical:

- start with pure noise
- watch the model remove noise step by step
- save every intermediate frame
- compare the early chaos with the final image
- control the number of denoising steps from the browser

This project is intentionally lightweight:

- it uses `google/ddpm-cifar10-32`
- it runs on CPU
- it avoids heavy image generation models
- it keeps the code simple enough to learn from

## What You Build

You end up with three things:

1. A tiny Python script that generates one image from noise.
2. A second script that saves every denoising step.
3. A custom HTML/CSS/JavaScript UI that keeps the screen focused on one thing at a time.

## Why This Project Exists

Diffusion models are easier to understand when you do not jump straight to the final picture.

If you only see the output image, it looks like the model is doing something mysterious.
If you see the intermediate frames, the process becomes much clearer:

- the first frame is just noise
- the middle frames start to form shapes
- the final frame becomes a recognizable image

That is the main teaching goal of this project.

## How The Project Works

The project is split into small steps on purpose.

Each step teaches one idea and proves one piece of the pipeline before the next step is added.

### Step 1: Set Up The Environment

What this step does:

- creates an isolated Python environment
- installs only the packages needed for the tutorial
- keeps everything CPU-only

Why this step matters:

- it prevents package conflicts
- it keeps the project lightweight for an 8 GB RAM machine
- it makes the rest of the steps easier to reproduce

How to do it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install diffusers transformers
```

### Step 2: Generate One Image From Noise

What this step does:

- loads `google/ddpm-cifar10-32`
- starts from random noise
- runs the diffusion model for a fixed number of denoising steps
- saves one final image

Why this step matters:

- it proves the model works on your machine
- it gives you the simplest possible working output
- it keeps the first success easy to understand

How to run it:

```bash
.venv/bin/python scripts/generate_single_image.py
```

What to expect:

- the image will be small because the model is trained for `32 x 32` images
- the result will usually look blurry or noisy at first glance
- that is fine, because the goal here is to prove the pipeline works

### Step 3: Capture Every Denoising Frame

What this step does:

- runs the same diffusion model again
- saves the initial pure noise image
- saves one image after every denoising step
- writes the final image to disk too

Why this step matters:

- this is where the tutorial becomes educational
- you can now see the model’s work, not just the result
- each saved frame shows one small move from chaos toward structure

How to run it:

```bash
.venv/bin/python scripts/generate_denoising_progression.py
```

What gets saved:

- `outputs/diffusion_progression/frames/frame_000.png`
- `outputs/diffusion_progression/frames/frame_001.png`
- ...
- `outputs/diffusion_progression/final_image.png`

### Step 4: Turn The Frames Into A Visual Summary

What this step does:

- reads the saved progression frames
- arranges them into a contact sheet
- puts the whole walk in one image

Why this step matters:

- it is easier to compare steps when they are visible at once
- the contact sheet makes the “noise to meaning” jump obvious
- it helps with teaching and debugging

How to run it:

```bash
.venv/bin/python scripts/visualize_denoising_progression.py
```

What gets saved:

- `outputs/diffusion_progression/contact_sheet.png`

### Step 5: Open The Custom Web UI

What this step does:

- starts a small Python HTTP server
- serves the custom HTML page
- runs the model when you press the button
- shows one screen for setup and one screen for the reveal
- lets you click the image to move forward one frame at a time

Why this step matters:

- it makes the project feel interactive
- it reduces the number of things on screen at once
- it keeps the browser simple and lets Python handle the model work

How to run it:

```bash
.venv/bin/python app.py
```

Then open:

- `http://127.0.0.1:8000`

What the page shows:

- a setup screen with one slider and one button
- a reveal screen with one image that advances by click
- a short status line while the model is running

### Step 6: Change The Number Of Denoising Steps

What this step does:

- adds a slider in the UI
- lets you choose between `10` and `100` denoising steps
- sends that value to Python when you click generate

Why this step matters:

- fewer steps are faster but rougher
- more steps are slower but usually cleaner and more structured
- this makes the tradeoff visible instead of theoretical

How to use it:

- move the slider before you click generate
- try a smaller value like `10`
- then try a larger value like `100`
- compare how long each run takes and how the frames evolve

What to look for:

- a low step count tends to keep the image rough and unfinished
- a high step count gives the model more chances to clean up the image
- the difference is easiest to notice in the middle frames
- the browser shows one starting noise frame plus one frame for each denoising step
- that means `10` steps gives you `11` frames, and `100` steps gives you `101` frames

## Project Files

- `scripts/generate_single_image.py` generates one final image from noise
- `scripts/generate_denoising_progression.py` saves every step in the diffusion walk
- `scripts/visualize_denoising_progression.py` builds the contact sheet
- `app.py` serves the browser UI and runs the generation API
- `diffusion_core.py` holds the shared diffusion logic
- `web/index.html` contains the page structure
- `web/styles.css` contains the visual design
- `web/app.js` connects the UI to Python

## What Happens When You Click Generate

1. The browser reads the step value from the slider.
2. The browser sends that step count to Python.
3. Python loads the cached DDPM pipeline.
4. Python starts from pure noise.
5. Python runs the denoising loop one step at a time.
6. Python saves every frame and the final image.
7. Python sends the list of frame URLs back to the browser.
8. The browser shows the first frame, then you click the image to move forward one step at a time.

## Run It Locally

```bash
.venv/bin/python app.py
```

Then visit:

```text
http://127.0.0.1:8000
```

## Notes

- The project is designed for CPU-only machines.
- The model is small on purpose.
- The images are tiny because `google/ddpm-cifar10-32` is trained for `32 x 32` outputs.
- More denoising steps usually mean more time on CPU.
- You may see a warning about `accelerate` the first time the model loads; the tutorial still works without it.
