# unnoise

Small demo: **prompt → noise → denoise step-by-step → image**, with a minimal browser UI.

**Primary way to run it:** [Google Colab](colab/unnoise.ipynb) (free GPU). One process (`app.py`) serves the UI **and** runs diffusion; you open it through Colab’s port preview — **nothing to install on your laptop.**

---

## Run on Google Colab (recommended)

1. Open the notebook: **`colab/unnoise.ipynb`**  
   - From GitHub: use Colab’s “Open in Colab” on that file, or upload the notebook to Colab.
2. **Runtime → Change runtime type → GPU**
3. Run all cells in order.

Last cell starts `app.py` and calls `output.serve_kernel_port_as_iframe(8000, …)` so you get the UI in the notebook (or a link Colab provides to port 8000).

**Optional env vars** (set in the last cell before `Popen` if you want):

| Variable | Meaning |
|----------|---------|
| `UI_MAX_FRAMES` | `all` = decode/save every step; or an integer to cap frames |
| `WARMUP_MODEL` | `1` (default) = load weights at startup; `0` = load on first Generate |
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
