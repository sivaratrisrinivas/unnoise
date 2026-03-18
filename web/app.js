const generateBtn = document.getElementById("generateBtn");
const stepSlider = document.getElementById("stepSlider");
const stepValue = document.getElementById("stepValue");
const statusText = document.getElementById("statusText");
const previewStage = document.getElementById("previewStage");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const selectedLabel = document.getElementById("selectedLabel");
const runMeta = document.getElementById("runMeta");
const frameName = document.getElementById("frameName");
const finalName = document.getElementById("finalName");
const contactSheet = document.getElementById("contactSheet");
const atlasPlaceholder = document.getElementById("atlasPlaceholder");
const frameGrid = document.getElementById("frameGrid");

let currentFrames = [];

function syncStepDisplay() {
  const steps = Number(stepSlider.value);
  const min = Number(stepSlider.min);
  const max = Number(stepSlider.max);
  const ratio = ((steps - min) / (max - min)) * 100;

  stepValue.textContent = `${steps} steps`;
  stepSlider.style.setProperty("--fill", `${ratio}%`);
}

function setBusy(isBusy) {
  document.body.classList.toggle("is-busy", isBusy);
  generateBtn.disabled = isBusy;
  stepSlider.disabled = isBusy;
}

function setStatus(message) {
  statusText.textContent = message;
}

function selectFrame(index) {
  if (!currentFrames.length) {
    return;
  }

  const frame = currentFrames[index];

  previewImage.src = frame.url;
  previewImage.hidden = false;
  previewStage.dataset.empty = "false";
  previewPlaceholder.hidden = true;
  selectedLabel.textContent = frame.label === "noise" ? "Noise" : `Denoising ${frame.label}`;
  frameName.textContent = frame.label;
  finalName.textContent = currentFrames[currentFrames.length - 1].label;

  for (const card of frameGrid.querySelectorAll(".frame-card")) {
    card.classList.toggle("is-active", Number(card.dataset.index) === index);
  }
}

function renderFrames(frames) {
  frameGrid.innerHTML = "";
  const fragment = document.createDocumentFragment();

  frames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frame-card";
    button.dataset.index = String(index);
    button.setAttribute("aria-label", `${frame.label}`);

    const thumb = document.createElement("img");
    thumb.className = "frame-thumb";
    thumb.src = frame.url;
    thumb.alt = frame.label;
    thumb.loading = "lazy";

    const label = document.createElement("div");
    label.className = "frame-label";
    label.innerHTML = `<strong>${frame.label}</strong><span>#${String(index).padStart(2, "0")}</span>`;

    button.appendChild(thumb);
    button.appendChild(label);
    button.addEventListener("click", () => selectFrame(index));
    fragment.appendChild(button);
  });

  frameGrid.appendChild(fragment);
}

function showEmptyState() {
  previewStage.dataset.empty = "true";
  previewImage.hidden = true;
  previewImage.removeAttribute("src");
  previewPlaceholder.hidden = false;
  selectedLabel.textContent = "Waiting for a run";
  runMeta.textContent = "Nothing generated yet.";
  frameName.textContent = "-";
  finalName.textContent = "-";
  contactSheet.hidden = true;
  atlasPlaceholder.hidden = false;
  frameGrid.innerHTML = "";
}

async function generate() {
  const steps = Number(stepSlider.value);
  setBusy(true);
  setStatus(`Loading the model and walking from noise to meaning with ${steps} steps...`);

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        steps,
        seed: 7,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Generation failed");
    }

    currentFrames = data.frames;
    renderFrames(currentFrames);
    selectFrame(currentFrames.length - 1);

    contactSheet.src = data.contact_sheet_url;
    contactSheet.hidden = false;
    atlasPlaceholder.hidden = true;

    runMeta.textContent = `${data.frame_count} frames · ${data.steps} steps · ${data.elapsed_seconds.toFixed(2)}s`;
    setStatus(`Run ${data.run_id} finished. Click any frame to inspect the path.`);
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

generateBtn.addEventListener("click", generate);
stepSlider.addEventListener("input", syncStepDisplay);
syncStepDisplay();
showEmptyState();
