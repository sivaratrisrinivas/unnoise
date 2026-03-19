# unnoise

`unnoise` is a CPU-only text-to-image demo that shows how a prompt becomes an image one denoising step at a time.

The goal is simple:

- type a prompt
- start from pure noise
- watch the image sharpen step by step
- keep the interface small enough to stay focused on the reveal

This version is built for a machine with:

- CPU only
- 8 GB RAM
- no GPU tricks

It uses `OFA-Sys/small-stable-diffusion-v0`, which is a real text-to-image model. That means the words you type are actually fed into the model. The old CIFAR-10 model was unconditional, so it could only generate random images from noise. This new model can follow prompts, which is what you wanted.

## What This Project Does

The app has one main idea: show the walk from noise to image instead of hiding it.

It does that in six small steps:

1. Set up a tiny CPU-only Python environment.
2. Generate one image from a text prompt.
3. Save every denoising step.
4. Turn the saved frames into a visual summary.
5. Open a minimal browser UI.
6. Let the user change how many denoising steps to run.

## Why This Model

I changed the model because the previous one was the wrong kind of diffusion model for your goal.

- the previous CIFAR-10 model was unconditional
- unconditional means it does not understand text prompts
- `OFA-Sys/small-stable-diffusion-v0` is text-to-image
- text-to-image means the prompt actually affects the final image

This model is also a better fit for your hardware than a full heavy model stack.

- it is smaller than standard Stable Diffusion v1.5
- it is documented as being nearly half the size of the baseline model
- the model card says it can run on CPU with OpenVINO in about 5 seconds at 10 steps on a Xeon-class CPU
- your machine will still be slower than that, but it is a realistic local target

## Step 1: Set Up The Environment

This stays intentionally small.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install diffusers transformers safetensors
```

What each line does:

- the virtual environment keeps the project isolated
- the PyTorch CPU wheel keeps everything CPU-only
- `diffusers` provides the diffusion pipelines
- `transformers` gives the text encoder pieces the model needs
- `safetensors` helps load the model weights cleanly

## Step 2: Generate One Image From A Prompt

This is the first real check that the model works.

Run:

```bash
.venv/bin/python scripts/generate_single_image.py --prompt "a red bicycle on a rainy street"
```

What happens:

- the script loads the prompt-based pipeline on CPU
- it starts from random noise
- it denoises for a fixed number of steps
- it saves one final image

Why this matters:

- it proves the model can follow text
- it proves the pipeline runs on your machine
- it gives you the simplest possible output before we add UI

You can change the prompt with `--prompt`.
You can change the number of steps with `--steps`.

## Step 3: Capture Every Denoising Frame

This step makes the process visible.

Run:

```bash
.venv/bin/python scripts/generate_denoising_progression.py --prompt "a red bicycle on a rainy street"
```

What happens:

- the script saves the starting noise frame
- it saves one frame after every denoising step
- it saves the final image at the end

Why this matters:

- you can see the model working instead of only seeing the result
- the middle frames show structure slowly appearing
- this is the core teaching moment of the project

The frames are saved under:

- `outputs/diffusion_progression/frames/`
- `outputs/diffusion_progression/final_image.png`

## Step 4: Build A Visual Summary

This turns the saved frames into one contact sheet.

Run:

```bash
.venv/bin/python scripts/visualize_denoising_progression.py
```

What happens:

- the script reads the saved frames from Step 3
- it lays them out in a grid
- it saves one summary image

Why this matters:

- a contact sheet makes the progression easy to scan
- it shows the noise-to-image walk at a glance
- it is useful for debugging and for teaching

The summary is saved at:

- `outputs/diffusion_progression/contact_sheet.png`

## Step 5: Open The Browser UI

This is the small HTML/CSS/JavaScript front end.

Run:

```bash
.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

What the browser shows:

- a setup screen with a prompt field and a step slider
- a loading screen while Python runs the model
- a reveal screen that auto-plays the denoising frames

Why this matters:

- the page stays focused on one action at a time
- the prompt is simple to change
- the reveal is the whole point, so the UI stays quiet

## Step 6: Change The Number Of Steps

The slider is now tuned for CPU prompt generation.

- minimum: `5`
- maximum: `20`
- default: `10`

What the slider changes:

- fewer steps are faster
- more steps are usually cleaner
- too many steps on CPU just makes the app slow without adding much value

What to expect:

- `5` steps gives you a rough, fast walk
- `10` steps is the default balance
- `20` steps is the cleanest option in this app

## What The Prompt Does

This is the important part.

The prompt is no longer just a seed.
It is passed into the model as text.

That means:

- different prompts produce different images
- the same prompt is reproducible here because the app keeps the random seed fixed
- the model still has limits, so it will not perfectly match every request

The model card for Stable Diffusion v1.5-style models also warns that:

- the model does not achieve perfect photorealism
- the model cannot render legible text well
- harder composition requests are still difficult

So the honest expectation is:

- prompt matters now
- exact prompt perfection still does not happen

## What Happens When You Click Generate

1. The browser reads the prompt and step count.
2. The browser sends both values to Python.
3. Python loads the cached pipeline, or downloads it if needed.
4. Python starts from random noise.
5. Python denoises step by step while storing the intermediate latents.
6. Python decodes each stored step into an image.
7. Python saves the frames and the final image.
8. The browser auto-plays the reveal.

## Project Files

- `scripts/generate_single_image.py` generates one final image from a prompt
- `scripts/generate_denoising_progression.py` saves every step in the diffusion walk
- `scripts/visualize_denoising_progression.py` builds the contact sheet
- `app.py` serves the browser UI and exposes the generation API
- `diffusion_core.py` holds the shared prompt-driven diffusion logic
- `web/index.html` contains the page structure
- `web/styles.css` contains the visual design
- `web/app.js` connects the UI to Python

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
- The old CIFAR-10 model has been removed from the app and deleted from the local Hugging Face cache.
- The output is still a diffusion sample, so it will not be perfect every time.
- More steps usually mean more waiting on CPU.
- The same prompt should produce the same walk here because the app uses a fixed seed behind the scenes.

## Sources

- [OFA-Sys/small-stable-diffusion-v0](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0)
- [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Diffusers pipeline callbacks](https://huggingface.co/docs/diffusers/main/using-diffusers/callback)
