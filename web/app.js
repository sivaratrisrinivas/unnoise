const setupScreen = document.getElementById("setupScreen");
const loadingScreen = document.getElementById("loadingScreen");
const revealScreen = document.getElementById("revealScreen");
const generateBtn = document.getElementById("generateBtn");
const generateBtnLabel = document.getElementById("generateBtnLabel");
const promptInput = document.getElementById("promptInput");
const inputLabel = document.getElementById("inputLabel");
const inputHint = document.getElementById("inputHint");
const stepSlider = document.getElementById("stepSlider");
const stepValue = document.getElementById("stepValue");
const setupMessage = document.getElementById("setupMessage");
const loadingModeLabel = document.getElementById("loadingModeLabel");
const loadingTitle = document.getElementById("loadingTitle");
const statusText = document.getElementById("statusText");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const frameLabel = document.getElementById("frameLabel");
const frameCounter = document.getElementById("frameCounter");
const resetBtn = document.getElementById("resetBtn");
const resultInputLabel = document.getElementById("resultInputLabel");
const resultSeedText = document.getElementById("resultSeedText");
const resultCompletion = document.getElementById("resultCompletion");
const resultPrompt = document.getElementById("resultPrompt");
const modeButtons = Array.from(document.querySelectorAll(".mode-option"));

let currentFrames = [];
let currentMode = "thought_completion";
let playbackToken = 0;

/** Milliseconds to show each frame during reveal autoplay (before advancing). */
const FRAME_AUTOPLAY_MS = 5000;

const MODE_CONFIGS = {
  thought_completion: {
    label: "Thought Completion",
    inputLabel: "Thought Fragment",
    defaultValue: "a man about to fall...",
    placeholder: "Type an incomplete scene",
    hint: "We reinterpret the fragment as a plausible next moment before diffusion starts.",
    loadingTitle: "Completing the implied scene.",
    loadingStatus: (steps) => `Conditioning ${steps} denoising steps on the next plausible moment...`,
    buttonLabel: "Complete the Thought",
    defaultCompletion: "The split second after a man begins to fall",
    defaultResolvedPrompt:
      "The split second after a man begins to fall. Show the consequence already unfolding in a single cinematic still...",
  },
  prompt: {
    label: "Direct Prompt",
    inputLabel: "Prompt",
    defaultValue: "a red bicycle on a rainy street",
    placeholder: "Describe the image you want",
    hint: "Use a full prompt when you want classic text-to-image generation.",
    loadingTitle: "Walking from words to image.",
    loadingStatus: (steps) => `Running ${steps} denoising steps...`,
    buttonLabel: "Generate Image",
    defaultCompletion: "Diffusion follows the prompt directly.",
    defaultResolvedPrompt: "a red bicycle on a rainy street",
  },
};

function getModeConfig(mode = currentMode) {
  return MODE_CONFIGS[mode];
}

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
  promptInput.disabled = isBusy;
  stepSlider.disabled = isBusy;
  resetBtn.disabled = isBusy;
  modeButtons.forEach((button) => {
    button.disabled = isBusy;
  });
}

function showScreen(screen) {
  setupScreen.hidden = screen !== "setup";
  loadingScreen.hidden = screen !== "loading";
  revealScreen.hidden = screen !== "reveal";
  document.body.dataset.view = screen;
}

function renderDefaultStoryCopy() {
  const config = getModeConfig();
  resultInputLabel.textContent = config.inputLabel;
  resultSeedText.textContent = promptInput.value.trim() || config.defaultValue;
  resultCompletion.textContent = config.defaultCompletion;
  resultPrompt.textContent = config.defaultResolvedPrompt;
}

function resetToSetup(message = "", isError = false) {
  playbackToken += 1;
  currentFrames = [];
  previewImage.hidden = true;
  previewImage.removeAttribute("src");
  previewPlaceholder.hidden = false;
  frameLabel.textContent = "noise";
  frameCounter.textContent = "1 / 1";
  resetBtn.hidden = true;
  renderDefaultStoryCopy();
  setupMessage.textContent = message;
  setupMessage.classList.toggle("is-error", isError);
  setBusy(false);
  showScreen("setup");
  promptInput.focus({ preventScroll: true });
}

function showLoading(steps) {
  const config = getModeConfig();
  loadingModeLabel.textContent = config.label;
  loadingTitle.textContent = config.loadingTitle;
  statusText.textContent = config.loadingStatus(steps);
  showScreen("loading");
}

function animateFrame() {
  previewImage.style.animation = "none";
  previewImage.offsetHeight;
  previewImage.style.animation = "frame-pop 220ms var(--ease-out) both";
}

function renderFrame(index) {
  const frame = currentFrames[index];

  previewPlaceholder.hidden = true;
  previewImage.hidden = false;
  previewImage.src = frame.url;
  previewImage.alt = frame.label === "noise" ? "Pure noise" : `${getModeConfig().label} ${frame.label}`;
  animateFrame();

  frameLabel.textContent = frame.label;
  frameCounter.textContent = `${index + 1} / ${currentFrames.length}`;
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function playFrames(token) {
  for (let index = 0; index < currentFrames.length; index += 1) {
    if (token !== playbackToken) {
      return false;
    }

    renderFrame(index);

    if (index < currentFrames.length - 1) {
      await sleep(FRAME_AUTOPLAY_MS);
    }
  }

  return token === playbackToken;
}

function showResetAction() {
  resetBtn.hidden = false;
  resetBtn.focus({ preventScroll: true });
}

function renderResultDetails(data) {
  resultInputLabel.textContent = data.input_label || getModeConfig().inputLabel;
  resultSeedText.textContent = data.seed_text || promptInput.value.trim() || getModeConfig().defaultValue;
  resultCompletion.textContent = data.completion_caption || getModeConfig().defaultCompletion;
  resultPrompt.textContent = data.resolved_prompt || data.prompt || getModeConfig().defaultResolvedPrompt;
}

function applyMode(mode, { preserveCustomValue = true } = {}) {
  const nextConfig = getModeConfig(mode);
  if (!nextConfig) {
    return;
  }

  const previousConfig = getModeConfig();
  const currentValue = promptInput.value.trim();
  const shouldReplaceValue =
    !preserveCustomValue || !currentValue || currentValue === previousConfig.defaultValue;

  currentMode = mode;
  document.body.dataset.mode = mode;

  modeButtons.forEach((button) => {
    const isActive = button.dataset.mode === mode;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });

  inputLabel.textContent = nextConfig.inputLabel;
  promptInput.placeholder = nextConfig.placeholder;
  inputHint.textContent = nextConfig.hint;
  generateBtnLabel.textContent = nextConfig.buttonLabel;
  loadingModeLabel.textContent = nextConfig.label;
  loadingTitle.textContent = nextConfig.loadingTitle;

  if (shouldReplaceValue) {
    promptInput.value = nextConfig.defaultValue;
  }

  if (document.body.dataset.view === "setup") {
    setupMessage.textContent = "";
    setupMessage.classList.remove("is-error");
  }

  renderDefaultStoryCopy();
}

async function generate() {
  const config = getModeConfig();
  const steps = Number(stepSlider.value);
  const seedText = promptInput.value.trim() || config.defaultValue;
  const token = ++playbackToken;

  setBusy(true);
  showLoading(steps);

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        steps,
        mode: currentMode,
        seed_text: seedText,
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

    renderResultDetails(data);
    showScreen("reveal");
    const completed = await playFrames(token);
    if (completed) {
      showResetAction();
    }
  } catch (error) {
    console.error(error);
    resetToSetup(error.message, true);
  } finally {
    setBusy(false);
  }
}

generateBtn.addEventListener("click", generate);
stepSlider.addEventListener("input", syncStepDisplay);
resetBtn.addEventListener("click", () => resetToSetup());
promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !generateBtn.disabled) {
    generate();
  }
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    applyMode(button.dataset.mode);
  });
});

syncStepDisplay();
applyMode("thought_completion", { preserveCustomValue: false });
resetToSetup();
