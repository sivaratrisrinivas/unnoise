const setupScreen = document.getElementById("setupScreen");
const loadingScreen = document.getElementById("loadingScreen");
const revealScreen = document.getElementById("revealScreen");
const generateBtn = document.getElementById("generateBtn");
const stepSlider = document.getElementById("stepSlider");
const stepValue = document.getElementById("stepValue");
const stepNote = document.getElementById("stepNote");
const statusText = document.getElementById("statusText");
const frameStage = document.getElementById("frameStage");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const frameLabel = document.getElementById("frameLabel");
const frameCounter = document.getElementById("frameCounter");

let currentFrames = [];
let currentIndex = 0;

function syncStepDisplay() {
  const steps = Number(stepSlider.value);
  const min = Number(stepSlider.min);
  const max = Number(stepSlider.max);
  const ratio = ((steps - min) / (max - min)) * 100;

  stepValue.textContent = String(steps);
  stepSlider.style.setProperty("--fill", `${ratio}%`);
}

function setBusy(isBusy) {
  generateBtn.disabled = isBusy;
  stepSlider.disabled = isBusy;
}

function showScreen(screen) {
  setupScreen.hidden = screen !== "setup";
  loadingScreen.hidden = screen !== "loading";
  revealScreen.hidden = screen !== "reveal";
  document.body.dataset.view = screen;
}

function resetToSetup(message = "The model runs on CPU and starts from noise.", isError = false) {
  currentFrames = [];
  currentIndex = 0;
  previewImage.hidden = true;
  previewImage.removeAttribute("src");
  previewPlaceholder.hidden = false;
  frameLabel.textContent = "noise";
  frameCounter.textContent = "1 / 1";
  frameStage.setAttribute("aria-label", "Generate a new walk");
  stepNote.textContent = message;
  stepNote.classList.toggle("is-error", isError);
  setBusy(false);
  showScreen("setup");
}

function showLoading(message) {
  statusText.textContent = message;
  showScreen("loading");
}

function animateFrame() {
  previewImage.style.animation = "none";
  previewImage.offsetHeight; // Force reflow so the next frame can animate in.
  previewImage.style.animation = "frame-pop 240ms var(--ease-out) both";
}

function renderFrame(index) {
  const frame = currentFrames[index];
  currentIndex = index;

  previewPlaceholder.hidden = true;
  previewImage.hidden = false;
  previewImage.src = frame.url;
  previewImage.alt = frame.label === "noise" ? "Pure noise" : `Denoising ${frame.label}`;
  animateFrame();

  frameLabel.textContent = frame.label;
  frameCounter.textContent = `${index + 1} / ${currentFrames.length}`;
  frameStage.setAttribute(
    "aria-label",
    index < currentFrames.length - 1
      ? `Advance to ${currentFrames[index + 1].label}`
      : "Return to the start"
  );
  showScreen("reveal");
  frameStage.focus({ preventScroll: true });
}

function advanceFrame() {
  if (!currentFrames.length) {
    return;
  }

  if (currentIndex < currentFrames.length - 1) {
    renderFrame(currentIndex + 1);
    return;
  }

  resetToSetup();
}

async function generate() {
  const steps = Number(stepSlider.value);
  setBusy(true);
  showLoading(`Running ${steps} denoising steps on CPU.`);

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
    if (!currentFrames.length) {
      throw new Error("No frames were returned.");
    }

    renderFrame(0);
  } catch (error) {
    console.error(error);
    resetToSetup(error.message, true);
  } finally {
    setBusy(false);
  }
}

generateBtn.addEventListener("click", generate);
stepSlider.addEventListener("input", syncStepDisplay);
frameStage.addEventListener("click", advanceFrame);
window.addEventListener("keydown", (event) => {
  if (document.body.dataset.view !== "reveal") {
    return;
  }

  if (event.key === "ArrowRight" || event.key === " " || event.key === "Enter") {
    event.preventDefault();
    advanceFrame();
  } else if (event.key === "Escape") {
    resetToSetup();
  }
});

syncStepDisplay();
resetToSetup();
