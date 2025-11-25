// ------------------------------
// Backend URL
// ------------------------------
// For local backend:
// const API_BASE = "http://127.0.0.1:8000";
// When deployed on HF Spaces, change to:
const API_BASE = "https://Ludvins-bayesipy-backend.hf.space";

// ------------------------------
// State & element refs
// ------------------------------
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const gridCountEl = document.getElementById("grid-count");

const methodSelect = document.getElementById("method-select");

const mlpEpochsInput = document.getElementById("mlp-epochs");
const mlpLrInput = document.getElementById("mlp-lr");
const mlpHiddenInput = document.getElementById("mlp-hidden");
const mlpDropoutInput = document.getElementById("mlp-dropout");

const optNoiseVarInput = document.getElementById("opt-noise-var");
const optIterationsInput = document.getElementById("opt-iterations");
const optLrInput = document.getElementById("opt-lr");
const optGridNInput = document.getElementById("opt-grid-n");

const optFmgpNumInducingInput = document.getElementById("opt-fmgp-num-inducing");

const optEllaSubsampleInput = document.getElementById("opt-ella-subsample");
const optEllaNeigsInput = document.getElementById("opt-ella-neigs");
const optEllaPriorInput = document.getElementById("opt-ella-prior");
const optEllaNoiseVarInput = document.getElementById("opt-ella-noise-var");

const optLLASubset = document.getElementById("opt-lla-subset");
const optLLAHessian = document.getElementById("opt-lla-hessian");

const optVallaNumInducingInput = document.getElementById("opt-valla-num-inducing");
const optVallaNoiseVarInput = document.getElementById("opt-valla-noise-var");
const optVallaPriorInput = document.getElementById("opt-valla-prior");
const optVallaInducingLocsSelect = document.getElementById("opt-valla-inducing-locs");

const optMfviSamplesInput = document.getElementById("opt-mfvi-samples");
const optMfviPriorInput = document.getElementById("opt-mfvi-prior");

const optSngpKernelScaleInput = document.getElementById("opt-sngp-kernel-scale");
const optSngpNrfInput = document.getElementById("opt-sngp-nrf");
const optSngpWeightDecayInput = document.getElementById("opt-sngp-weight-decay");

const btnClear = document.getElementById("btn-clear");
const btnTrainMLP = document.getElementById("btn-train-mlp");
const btnPredictMLP = document.getElementById("btn-predict-mlp");
const btnRunMethod = document.getElementById("btn-run-method");

const blockFmgp = document.getElementById("block-fmgp");
const blockElla = document.getElementById("block-ella");
const blockLla = document.getElementById("block-lla");
const blockValla = document.getElementById("block-valla");

const blockMfvi = document.getElementById("block-mfvi");
const blockSngp = document.getElementById("block-sngp");

const optionsElla = document.getElementById("options-ella");
const optionsValla = document.getElementById("options-valla");

let dataX = [];
let dataY = [];

let lastGridX = [];
let lastMean = [];
let lastLower = [];
let lastUpper = [];
let lastSamples = null; // for future sample-based methods

const X_MIN = -5;
const X_MAX = 5;
const Y_MIN = -5;
const Y_MAX = 5;
const Y_MIN_CLICK = -5;
const Y_MAX_CLICK = 5;

function readFloat(input, fallback) {
  const v = parseFloat(input.value);
  return Number.isFinite(v) ? v : fallback;
}

function readInt(input, fallback) {
  const v = parseInt(input.value, 10);
  return Number.isFinite(v) ? v : fallback;
}

// ------------------------------
// Method-specific options visibility
// ------------------------------
function updateMethodOptionsVisibility() {
  const method = methodSelect.value;

  if (blockFmgp) blockFmgp.style.display = "none";
  if (blockValla) blockValla.style.display = "none";
  if (blockMfvi) blockMfvi.style.display = "none";
  if (blockSngp) blockSngp.style.display = "none";
  if (blockElla) blockElla.style.display = "none";
  if (blockLla) blockLla.style.display = "none";
  if (blockValla) blockValla.style.display = "none";

  if (method === "fmgp") {
    blockFmgp.style.display = "block";
  } else if (method === "ella") {
    blockElla.style.display = "block";
    optionsElla.style.display = "grid";
  } else if (method === "valla") {
    blockValla.style.display = "block";
    optionsValla.style.display = "grid";
  } else if (method === "mfvi") {
    blockMfvi.style.display = "block";
  } else if (method === "lla") {
    blockLla.style.display = "block";
  } else if (method === "sngp") {
    blockSngp.style.display = "block";
  }
}

methodSelect.addEventListener("change", updateMethodOptionsVisibility);

// ------------------------------
// Canvas interaction
// ------------------------------
canvas.addEventListener("click", (evt) => {
  const rect = canvas.getBoundingClientRect();
  const px = evt.clientX - rect.left;
  const py = evt.clientY - rect.top;
  const w = rect.width;
  const h = rect.height;

  const x = X_MIN + (px / w) * (X_MAX - X_MIN);
  const y = Y_MAX_CLICK - (py / h) * (Y_MAX_CLICK - Y_MIN_CLICK);

  dataX.push(x);
  dataY.push(y);

  lastGridX = [];
  lastMean = [];
  lastLower = [];
  lastUpper = [];
  lastSamples = null;
  setStatus("Added point (" + x.toFixed(2) + ", " + y.toFixed(2) + ")");
  render();
});

btnClear.addEventListener("click", () => {
  dataX = [];
  dataY = [];
  lastGridX = [];
  lastMean = [];
  lastLower = [];
  lastUpper = [];
  lastSamples = null;
  setStatus("Cleared points.");
  render();
});

// ------------------------------
// Rendering (axes fixed)
// ------------------------------
function render() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const yMin = Y_MIN;
  const yMax = Y_MAX;

  function xToCanvas(x) {
    return ((x - X_MIN) / (X_MAX - X_MIN)) * w;
  }
  function yToCanvas(y) {
    const yy = Math.max(yMin, Math.min(yMax, y));
    return h - ((yy - yMin) / (yMax - yMin)) * h;
  }

  // Axes
  ctx.save();
  ctx.strokeStyle = "rgba(148,163,184,0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const x0 = xToCanvas(0);
  ctx.moveTo(x0, 0);
  ctx.lineTo(x0, h);
  const y0 = yToCanvas(0);
  ctx.moveTo(0, y0);
  ctx.lineTo(w, y0);
  ctx.stroke();
  ctx.restore();

  // Draw samples or distribution
  if (lastSamples && lastSamples.length > 0 && lastGridX.length > 0) {
    ctx.save();
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = "rgba(56,189,248,0.85)";
    for (const sample of lastSamples) {
      if (sample.length !== lastGridX.length) continue;
      ctx.beginPath();
      for (let i = 0; i < lastGridX.length; i++) {
        const cx = xToCanvas(lastGridX[i]);
        const cy = yToCanvas(sample[i]);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }
    ctx.restore();
  } else if (lastMean.length > 0 && lastGridX.length > 0) {
    if (lastLower.length === lastMean.length && lastUpper.length === lastMean.length) {
      ctx.save();
      ctx.fillStyle = "rgba(59,130,246,0.18)";
      ctx.beginPath();
      for (let i = 0; i < lastGridX.length; i++) {
        const cx = xToCanvas(lastGridX[i]);
        const cy = yToCanvas(lastUpper[i]);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }
      for (let i = lastGridX.length - 1; i >= 0; i--) {
        const cx = xToCanvas(lastGridX[i]);
        const cy = yToCanvas(lastLower[i]);
        ctx.lineTo(cx, cy);
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    ctx.save();
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < lastGridX.length; i++) {
      const cx = xToCanvas(lastGridX[i]);
      const cy = yToCanvas(lastMean[i]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Points
  ctx.save();
  ctx.fillStyle = "#f97316";
  for (let i = 0; i < dataX.length; i++) {
    const cx = xToCanvas(dataX[i]);
    const cy = yToCanvas(dataY[i]);
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function setStatus(msg, isError = false) {
  if (!statusEl) return;
  if (isError) {
    statusEl.innerHTML = '<span class="error">' + msg + "</span>";
  } else {
    statusEl.textContent = msg;
  }
}

// ------------------------------
// Backend calls
// ------------------------------
function buildMLPConfig() {
  return {
    epochs: readInt(mlpEpochsInput, 200),
    lr: readFloat(mlpLrInput, 0.01),
    hidden_dim: readInt(mlpHiddenInput, 32),
    dropout_rate: readFloat(mlpDropoutInput, 0.0)
  };
}

function buildOptions(method, nGrid) {
  const opts = {};
  const noiseVar = readFloat(optNoiseVarInput, 0.01);
  const iterations = readInt(optIterationsInput, 2000);
  const lr = readFloat(optLrInput, 0.001);

  if (method === "fmgp") {
    opts.kernel = "RBF";
    opts.inducing_locations = "kmeans";
    opts.num_inducing = optFmgpNumInducingInput.value;
    opts.noise_variance = noiseVar;
    opts.iterations = iterations;
    opts.lr = lr;
    opts.seed = 0;
  } else if (method === "lla") {
    opts.subset = optLLASubset.value;
    opts.hessian = optLLAHessian.value;
  } else if (method === "ella") {
    opts.ella_subsample = readInt(optEllaSubsampleInput, 100);
    opts.ella_n_eigenvalues = readInt(optEllaNeigsInput, 10);
    opts.ella_prior = readFloat(optEllaPriorInput, 1.0);
    opts.ella_noise_variance = readFloat(optEllaNoiseVarInput, 1.0);
    opts.seed = 0;
  } else if (method === "valla") {
    opts.valla_noise_variance = readFloat(optVallaNoiseVarInput, 1.0);
    opts.valla_prior = readFloat(optVallaPriorInput, 1.0);
    opts.valla_inducing_locations = optVallaInducingLocsSelect.value;
    opts.valla_num_inducing = optVallaNumInducingInput.value;
    opts.iterations = iterations;
    opts.lr = lr;
    opts.seed = 0;
  } else if (method === "mfvi") {
    opts.noise_variance = noiseVar;
    opts.prior = readFloat(optMfviPriorInput, 1.0);
    opts.n_samples = readInt(optMfviSamplesInput, 20);
    opts.iterations = iterations;
    opts.seed = 0;
  } else if (method === "sngp") {
    opts.kernel_scale = readFloat(optSngpKernelScaleInput, 1.0);
    opts.n_random_features = readInt(optSngpNrfInput, 256);
    opts.noise_variance = noiseVar;
    opts.gp_output_bias = 0.0;
    opts.layer_norm_eps = 1e-6;
    opts.n_power_iterations = 1;
    opts.scale_random_features = false;
    opts.normalize_input = false;
    opts.gp_cov_momentum = 0.999;
    opts.gp_cov_ridge_penalty = 0.001;
    opts.iterations = iterations;
    opts.lr = lr;
    opts.weight_decay = readFloat(optSngpWeightDecayInput, 0.1);
    opts.seed = 0;
  }

  return opts;
}

async function callBackendTrainMLP() {
  if (dataX.length < 2) {
    setStatus("Need at least 2 points to train the MLP.", true);
    return;
  }
  const nGrid = readInt(optGridNInput, 200);
  gridCountEl.textContent = String(nGrid);

  const payload = {
    x: dataX,
    y: dataY,
    method: "map",
    mlp: buildMLPConfig(),
    options: {},
    n_grid: nGrid,
    n_samples: 20
  };

  try {
    btnTrainMLP.disabled = true;
    setStatus("Training MLP on backend…");
    const resp = await fetch(API_BASE + "/api/train_mlp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      throw new Error("Backend error: " + resp.status);
    }
    const result = await resp.json();
    handlePredictionResponse(result, "Trained MLP (MAP) and stored it on backend.");
  } catch (err) {
    console.error(err);
    setStatus("MLP training failed: " + err.message, true);
  } finally {
    btnTrainMLP.disabled = false;
  }
}

async function callBackendPredictMLP() {
  try {
    const nGrid = readInt(optGridNInput, 200);
    gridCountEl.textContent = String(nGrid);
    const payload = { n_grid: nGrid };
    btnPredictMLP.disabled = true;
    setStatus("Requesting prediction from stored MLP…");
    const resp = await fetch(API_BASE + "/api/predict_mlp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      throw new Error("Backend error: " + resp.status);
    }
    const result = await resp.json();
    handlePredictionResponse(result, "Predicted with stored MLP (no retraining).");
  } catch (err) {
    console.error(err);
    setStatus("Stored MLP prediction failed: " + err.message, true);
  } finally {
    btnPredictMLP.disabled = false;
  }
}

async function callBackendRunMethod() {
  if (dataX.length < 2) {
    setStatus("Need at least 2 points to run a method.", true);
    return;
  }
  const method = methodSelect.value;
  const nGrid = readInt(optGridNInput, 200);
  gridCountEl.textContent = String(nGrid);

  const payload = {
    x: dataX,
    y: dataY,
    method: method,
    mlp: buildMLPConfig(),
    options: buildOptions(method, nGrid),
    n_grid: nGrid,
    n_samples: 20
  };

  try {
    btnRunMethod.disabled = true;
    setStatus("Running " + method.toUpperCase() + " on backend…");
    const resp = await fetch(API_BASE + "/api/predict_1d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      throw new Error("Backend error: " + resp.status);
    }
    const result = await resp.json();
    handlePredictionResponse(
      result,
      "Method " + method.toUpperCase() + " finished."
    );
  } catch (err) {
    console.error(err);
    setStatus("Method call failed: " + err.message, true);
  } finally {
    btnRunMethod.disabled = false;
  }
}

function handlePredictionResponse(result, okMessage) {
  lastGridX = result.grid_x || [];
  if (result.kind === "samples") {
    lastSamples = result.samples || [];
    lastMean = [];
    lastLower = [];
    lastUpper = [];
    setStatus(okMessage + " (samples mode)");
  } else {
    lastSamples = null;
    lastMean = result.mean || [];
    lastLower = result.lower || [];
    lastUpper = result.upper || [];
    setStatus(okMessage);
  }
  render();
}

// ------------------------------
// Wire up buttons & init
// ------------------------------
btnTrainMLP.addEventListener("click", () => {
  callBackendTrainMLP();
});

btnPredictMLP.addEventListener("click", () => {
  callBackendPredictMLP();
});

btnRunMethod.addEventListener("click", () => {
  callBackendRunMethod();
});

updateMethodOptionsVisibility();
render();
