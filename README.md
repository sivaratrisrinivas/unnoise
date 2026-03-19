# unnoise

Small demo: **fragment or prompt -> conditioned scene -> noise -> denoise step-by-step -> image**, with a minimal browser UI.

It still supports the original direct prompt flow, and now it also has a **Thought Completion** mode:

- Type a fragment like `a man about to fall...`
- The app turns that into a "next plausible moment" conditioning prompt
- Diffusion starts from noise and reveals the implied future scene step by step

**Primary way to run it:** Google Colab (free GPU). One process (`app.py`) serves the UI **and** runs diffusion; you open it through Colab’s port preview — **nothing to install on your laptop.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sivaratrisrinivas/unnoise/blob/main/colab/unnoise.ipynb)

GitHub **does not** show an “Open in Colab” control on the notebook file page — that badge above is how you get one-click open. Alternatives:

- Paste this URL: `https://colab.research.google.com/github/sivaratrisrinivas/unnoise/blob/main/colab/unnoise.ipynb`
- In Colab: **File → Open notebook →** tab **GitHub** → org/repo `sivaratrisrinivas/unnoise` → open `colab/unnoise.ipynb`

---

## Run on Google Colab (recommended)

1. Open the notebook via the badge at the top of this README (or one of the alternatives above).
2. **Runtime → Change runtime type → GPU**
3. Run all cells in order.

Last cell starts `app.py`, waits for port 8000, **runs `serve_kernel_port_as_iframe` first** (registers the tunnel so `*.prod.colab.dev` does not 404), then prints a **proxy URL** for an optional new tab. **Open new-tab links in the same browser where Colab is logged in** — pasting the URL into another browser/device usually returns HTTP 404. **Do not open `localhost:8000` on your own PC** (that is your machine, not the VM). If the server never binds, the cell prints `/tmp/unnoise_app.log`.

**Why you might have seen a blank iframe:** `WARMUP_MODEL=1` runs a full model download/load **before** `app.py` binds to port 8000; a short `sleep(10)` then embeds an empty port. The bundled notebook sets `WARMUP_MODEL=0` so the server starts immediately; the **first Generate** pays the Hugging Face download.

**Optional env vars** (edit the last notebook cell if you want):

| Variable | Meaning |
|----------|---------|
| `UI_MAX_FRAMES` | `all` = decode/save every step; or an integer to cap frames |
| `WARMUP_MODEL` | Notebook uses `0` for reliable iframe. `1` = preload at startup (can take many minutes first time). |
| `DEVICE` | `cuda` on Colab; omit to auto-detect |
| `DEFAULT_MODE` | Default `mode` when the client omits it: `prompt` or `thought_completion` (default: `prompt`) |

After you change this repo, **push to GitHub** and in Colab **re-run the clone cell** or `git pull` inside `/content/unnoise` so the notebook picks up the latest `web/`, `app.py`, and `diffusion_core.py`. No extra pip packages are required for Thought Completion.

**HF downloads:** if Hugging Face asks for auth, set `HF_TOKEN` in Colab (`os.environ["HF_TOKEN"] = "..."`) or add a secret; you can also use a repo-root `.env.local` with `HF_TOKEN=...` (not committed; see `.gitignore`).

---

## What `git clone` copies (and what it does not)

`git clone` downloads **every file tracked in git** for this repo — that’s the whole app:

| Path | Why it exists |
|------|----------------|
| `app.py` | HTTP server: serves `web/`, `POST /api/generate`, `GET /outputs/...`, returns interpreted completion metadata |
| `diffusion_core.py` | Loads the diffusers pipeline, conditions prompt fragments, runs denoising, saves frames |
| `web/index.html`, `web/styles.css`, `web/app.js` | Browser UI with `Thought Completion` and `Direct Prompt` modes |
| `scripts/*.py` | CLI helpers (single image, full progression, contact sheet) |
| `README.md`, `.gitignore` | Docs + ignore rules |

**Not** included (on purpose, via `.gitignore`):

- `.venv/` — you don’t commit virtualenvs; Colab installs packages in the notebook.
- `outputs/` — generated images; created at runtime.
- `.env.local` — optional secrets; never commit.

So: **clone = full source tree.** You don’t cherry-pick individual files; Colab needs at least `app.py`, `diffusion_core.py`, and `web/` for the interactive app.

---

## Optional: CLI scripts (same clone, any machine with Python)

After `pip install` (CPU or CUDA torch + `diffusers transformers safetensors pillow`):

```bash
python scripts/generate_single_image.py --prompt "a red bicycle on a rainy street"
python scripts/generate_denoising_progression.py --prompt "a red bicycle on a rainy street"
python scripts/generate_single_image.py --mode thought_completion --prompt "a man about to fall..."
python scripts/generate_denoising_progression.py --mode thought_completion --prompt "a glass about to shatter"
python scripts/visualize_denoising_progression.py
```

---

## What happens when you click Generate

1. Browser sends `seed_text`, `mode`, and `steps` to `POST /api/generate`.
2. Python resolves the input into a conditioning prompt:
   - `mode=prompt`: use the text directly
   - `mode=thought_completion`: rewrite the fragment as the immediate next plausible moment
3. Python loads the pipeline (cached after first time).
4. Diffusion runs for N steps; intermediate latents are kept per your `UI_MAX_FRAMES` setting.
5. Each kept step is VAE-decoded and saved under `outputs/ui_runs/<run_id>/`.
6. JSON returns frame URLs plus the interpreted completion text; the UI loads everything from the same host.
7. On the **reveal** screen, autoplay advances frames every **5 seconds** so each step is easy to see. Adjust `FRAME_AUTOPLAY_MS` in [`web/app.js`](web/app.js) (milliseconds) if you want a shorter or longer hold.

### `POST /api/generate` (for custom clients)

**JSON body**

| Field | Required | Notes |
|-------|----------|--------|
| `steps` | no | Denoising steps (default from server; UI uses 5–20) |
| `mode` | no | `prompt` or `thought_completion`; falls back to `DEFAULT_MODE` |
| `seed_text` | no | Main text field; empty uses built-in defaults per mode |
| `prompt` | no | Alias for `seed_text` (backwards compatible) |

**Response** (in addition to `frames`, `final_image_url`, timing fields): `mode`, `mode_label`, `seed_text`, `input_label`, `completion_caption`, `resolved_prompt`, and `prompt` (same as `resolved_prompt`).

---

## Thought Completion Mode

This is the new "incomplete thought" path in the app.

Example:

- Input fragment: `a man about to fall...`
- Completion caption: `The split second after a man begins to fall`
- Diffusion prompt: that caption plus visual instructions for a single coherent cinematic still

This is not a separate model. The project still uses the same Stable Diffusion pipeline, but now it adds a small prompt-conditioning layer in [`diffusion_core.py`](diffusion_core.py) before sampling when you choose `Thought Completion`.

The current conditioning layer is intentionally simple and local:

- it detects incomplete cues like `about to`, `on the verge of`, and `almost`
- it reframes them as the next believable moment
- it sends that resolved prompt into the diffusion run and exposes the interpretation in the UI

That makes the demo feel less like "generate an image from scratch" and more like "finish the scene I was hinting at."

---

## Model

- **Hugging Face:** [`OFA-Sys/small-stable-diffusion-v0`](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0)

UI modes:

- `Thought Completion` is the default browser experience
- `Direct Prompt` preserves the original prompt-to-image flow

Step slider in the UI: **5–20** (default **10**). Fixed seed in code for reproducibility.

---

## Sources

- [OFA-Sys/small-stable-diffusion-v0](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0)
- [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Diffusers pipeline callbacks](https://huggingface.co/docs/diffusers/main/using-diffusers/callback)
