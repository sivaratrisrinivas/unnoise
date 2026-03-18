const setupScreen = document.getElementById("setupScreen");
const loadingScreen = document.getElementById("loadingScreen");
const revealScreen = document.getElementById("revealScreen");
const generateBtn = document.getElementById("generateBtn");
const seedInput = document.getElementById("seedInput");
const stepSlider = document.getElementById("stepSlider");
const stepValue = document.getElementById("stepValue");
const setupMessage = document.getElementById("setupMessage");
const statusText = document.getElementById("statusText");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const frameLabel = document.getElementById("frameLabel");
const frameCounter = document.getElementById("frameCounter");

let currentFrames = [];

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
  seedInput.disabled = isBusy;
  stepSlider.disabled = isBusy;
}

function showScreen(screen) {
  setupScreen.hidden = screen !== "setup";
  loadingScreen.hidden = screen !== "loading";
  revealScreen.hidden = screen !== "reveal";
  document.body.dataset.view = screen;
}

function resetToSetup(message = "Any text works. It becomes the seed for the noise.", isError = false) {
  currentFrames = [];
  previewImage.hidden = true;
  previewImage.removeAttribute("src");
  previewPlaceholder.hidden = false;
  frameLabel.textContent = "noise";
  frameCounter.textContent = "1 / 1";
  setupMessage.textContent = message;
  setupMessage.classList.toggle("is-error", isError);
  setBusy(false);
  showScreen("setup");
}

function showLoading(message) {
  statusText.textContent = message;
  statusText.classList.remove("is-error");
  showScreen("loading");
}

function animateFrame() {
  previewImage.style.animation = "none";
  previewImage.offsetHeight; // Force a reflow so the new frame can animate in.
  previewImage.style.animation = "frame-pop 220ms var(--ease-out) both";
}

function frameDelay(totalFrames) {
  return Math.max(55, Math.min(140, Math.round(9000 / totalFrames)));
}

function renderFrame(index) {
  const frame = currentFrames[index];

  previewPlaceholder.hidden = true;
  previewImage.hidden = false;
  previewImage.src = frame.url;
  previewImage.alt = frame.label === "noise" ? "Pure noise" : `Denoising ${frame.label}`;
  animateFrame();

  frameLabel.textContent = frame.label;
  frameCounter.textContent = `${index + 1} / ${currentFrames.length}`;
}

async function playFrames() {
  const delay = frameDelay(currentFrames.length);

  for (let index = 0; index < currentFrames.length; index += 1) {
    renderFrame(index);

    if (index < currentFrames.length - 1) {
      await new Promise((resolve) => {
        window.setTimeout(resolve, delay);
      });
    }
  }
}

async function generate() {
  const steps = Number(stepSlider.value);
  const seedText = seedInput.value.trim();
  const phrase = seedText || "unnoise";

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
        seed_text: phrase,
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

    showScreen("reveal");
    await playFrames();
  } catch (error) {
    console.error(error);
    resetToSetup(error.message, true);
  } finally {
    setBusy(false);
  }
}

generateBtn.addEventListener("click", generate);
stepSlider.addEventListener("input", syncStepDisplay);

syncStepDisplay();
resetToSetup();
