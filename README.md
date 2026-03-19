# unnoise

Small demo: **prompt → noise → denoise step-by-step → image**, with a minimal browser UI.

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

Last cell starts `app.py`, **waits until port 8000 accepts connections**, then calls `output.serve_kernel_port_as_window(8000, …)` so the UI opens in a **new browser tab** (not only inside the notebook). If popups are blocked, allow them for Colab or switch to `serve_kernel_port_as_iframe` in the notebook. If the server never binds, it prints the tail of `/tmp/unnoise_app.log`.

**Why you might have seen a blank iframe:** `WARMUP_MODEL=1` runs a full model download/load **before** `app.py` binds to port 8000; a short `sleep(10)` then embeds an empty port. The bundled notebook sets `WARMUP_MODEL=0` so the server starts immediately; the **first Generate** pays the Hugging Face download.

**Optional env vars** (edit the last notebook cell if you want):

| Variable | Meaning |
|----------|---------|
| `UI_MAX_FRAMES` | `all` = decode/save every step; or an integer to cap frames |
| `WARMUP_MODEL` | Notebook uses `0` for reliable iframe. `1` = preload at startup (can take many minutes first time). |
| `DEVICE` | `cuda` on Colab; omit to auto-detect |

**HF downloads:** if Hugging Face asks for auth, set `HF_TOKEN` in Colab (`os.environ["HF_TOKEN"] = "..."`) or add a secret; you can also use a repo-root `.env.local` with `HF_TOKEN=...` (not committed; see `.gitignore`).

---

## What `git clone` copies (and what it does not)

`git clone` downloads **every file tracked in git** for this repo — that’s the whole app:

| Path | Why it exists |
|------|----------------|
| `app.py` | HTTP server: serves `web/`, `POST /api/generate`, `GET /outputs/...` |
| `diffusion_core.py` | Loads the diffusers pipeline, runs denoising, saves frames |
| `web/index.html`, `web/styles.css`, `web/app.js` | Browser UI |
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
python scripts/visualize_denoising_progression.py
```

---

## What happens when you click Generate

1. Browser sends prompt + steps to `POST /api/generate`.
2. Python loads the pipeline (cached after first time).
3. Diffusion runs for N steps; intermediate latents are kept per your `UI_MAX_FRAMES` setting.
4. Each kept step is VAE-decoded and saved under `outputs/ui_runs/<run_id>/`.
5. JSON lists frame URLs; the UI loads them from the same host.

---

## Model

- **Hugging Face:** [`OFA-Sys/small-stable-diffusion-v0`](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0)

Step slider in the UI: **5–20** (default **10**). Fixed seed in code for reproducibility.

---

## Sources

- [OFA-Sys/small-stable-diffusion-v0](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0)
- [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Diffusers pipeline callbacks](https://huggingface.co/docs/diffusers/main/using-diffusers/callback)
