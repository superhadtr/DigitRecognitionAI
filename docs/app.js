// Digit recognition: drawing -> 28x28 preprocessing -> forward pass -> percentages
const pad = document.getElementById("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const seen = document.getElementById("seen");
const seenCtx = seen.getContext("2d");
const statusEl = document.getElementById("status");
const predictBtn = document.getElementById("predictBtn");
const digitEl = document.getElementById("digit");
const confEl = document.getElementById("conf");
const barsEl = document.getElementById("bars");

const N = 280;
let MODEL = null;
let drawing = false;
let last_input = null;

// --- canvas setup: black background (MNIST standard) ---
function resetPad() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, N, N);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
resetPad();

// --- drawing ---
function pos(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * (N / r.width), (e.clientY - r.top) * (N / r.height)];
}
pad.addEventListener("pointerdown", (e) => {
  drawing = true;
  pad.setPointerCapture(e.pointerId);
  const [x, y] = pos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 0.1, y + 0.1);
  ctx.stroke();
});
pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const [x, y] = pos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});
function stopDraw() {
  if (!drawing) return;
  drawing = false;
  ctx.beginPath(); // close the unfinished stroke
}
pad.addEventListener("pointerup", stopDraw);
pad.addEventListener("pointercancel", stopDraw);

document.getElementById("clearBtn").addEventListener("click", () => {
  resetPad();
  digitEl.textContent = "–";
  confEl.textContent = "";
  document.querySelectorAll(".bar .fill").forEach((f) => (f.style.width = "0%"));
  document.querySelectorAll(".bar .pct").forEach((p) => (p.textContent = "%0.0"));
  document.querySelectorAll(".bar").forEach((b) => b.classList.remove("top"));
  seenCtx.fillStyle = "#000";
  seenCtx.fillRect(0, 0, 28, 28);
  const hCtx = document.getElementById("heatmap")?.getContext("2d");
  if (hCtx) {
    hCtx.fillStyle = "#000";
    hCtx.fillRect(0, 0, 28, 28);
  }
  document.getElementById("correction").style.display = "none";
  last_input = null;
});
predictBtn.addEventListener("click", predict);

// --- feedback loop ---
document.querySelectorAll(".corrBtn").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    if (!last_input) return;
    const target = parseInt(e.target.dataset.val);
    // Use a very very small learning rate so it behaves like 1 out of 1000 examples
    train_step(last_input, target, 0.001);
    
    // Re-predict to show updated results
    predict();
    statusEl.textContent = `✅ I nudged my weights towards ${target}. (It learns slowly to not forget other digits!)`;
  });
});

document.getElementById("downloadBtn").addEventListener("click", () => {
  if (!last_input) return;
  const canvas = document.createElement("canvas");
  canvas.width = 28;
  canvas.height = 28;
  const dCtx = canvas.getContext("2d");
  const imgData = dCtx.createImageData(28, 28);
  for (let i=0; i<784; i++) {
    const val = Math.floor(last_input[i] * 255);
    imgData.data[i*4] = val;
    imgData.data[i*4+1] = val;
    imgData.data[i*4+2] = val;
    imgData.data[i*4+3] = 255;
  }
  dCtx.putImageData(imgData, 0, 0);
  
  const link = document.createElement("a");
  link.download = `my_digit_${digitEl.textContent || "unknown"}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
});

// --- percentage bars ---
for (let d = 0; d <= 9; d++) {
  const row = document.createElement("div");
  row.className = "bar";
  row.id = "bar" + d;
  row.innerHTML = `<span class="lbl">${d}</span>
    <div class="track"><div class="fill"></div></div>
    <span class="pct">%0.0</span>`;
  barsEl.appendChild(row);
}

// --- load model ---
fetch("weights.json")
  .then((r) => {
    if (!r.ok) throw new Error("weights.json not found");
    return r.json();
  })
  .then((w) => {
    MODEL = w;
    predictBtn.disabled = false;
    const acc = (w.test_accuracy * 100).toFixed(2);
    document.getElementById("arch").textContent =
      `Model: feedforward network (${w.arch.join(" → ")}) trained on MNIST.`;
    statusEl.textContent = `Model ready ✅ (test accuracy: %${acc}) — draw and release, it predicts automatically.`;
  })
  .catch((err) => {
    statusEl.textContent = "Failed to load model: " + err.message;
  });

// --- 28x28 preprocessing (same logic as the Python code) ---
function preprocess() {
  const img = ctx.getImageData(0, 0, N, N).data;
  let x0 = N, y0 = N, x1 = -1, y1 = -1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (img[(y * N + x) * 4] > 20) { // is there a white pixel
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null; // empty canvas

  const m = 14;
  x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m);
  x1 = Math.min(N - 1, x1 + m); y1 = Math.min(N - 1, y1 + m);
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1;

  // put the cropped region on a temporary canvas
  const crop = document.createElement("canvas");
  crop.width = cw; crop.height = ch;
  crop.getContext("2d").putImageData(ctx.getImageData(x0, y0, cw, ch), 0, 0);

  // keep aspect ratio, fit into 20x20, center on 28x28
  const s = 20 / Math.max(cw, ch);
  const w = Math.max(1, Math.round(cw * s)), h = Math.max(1, Math.round(ch * s));
  seenCtx.fillStyle = "#000";
  seenCtx.fillRect(0, 0, 28, 28);
  seenCtx.imageSmoothingEnabled = true;
  seenCtx.imageSmoothingQuality = "high";
  seenCtx.drawImage(crop, Math.floor((28 - w) / 2), Math.floor((28 - h) / 2), w, h);

  const px = seenCtx.getImageData(0, 0, 28, 28).data;
  const input = new Array(784);
  for (let i = 0; i < 784; i++) input[i] = px[i * 4] / 255;
  return input;
}

// --- forward pass: x -> relu(x*W1+b1) -> relu(h*W2+b2) -> softmax ---
function relu(a) { return a.map((v) => (v > 0 ? v : 0)); }
function matvec(x, W, b) { // W: [in][out]
  const out = new Array(W[0].length).fill(0);
  for (let i = 0; i < W.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = W[i];
    for (let j = 0; j < row.length; j++) out[j] += xi * row[j];
  }
  for (let j = 0; j < out.length; j++) out[j] += b[j];
  return out;
}
function forward_with_cache(input) {
  let h = input;
  const cache = [{ h: h.slice() }];
  const L = MODEL.layers;
  for (let l = 0; l < L.length; l++) {
    h = matvec(h, L[l].W, L[l].b);
    let z = h.slice();
    if (l < L.length - 1) h = relu(h);
    cache.push({ z: z, h: h.slice() });
  }
  const mx = Math.max(...h);
  const e = h.map((v) => Math.exp(v - mx));
  const sum = e.reduce((a, b) => a + b, 0);
  return { probs: e.map((v) => v / sum), cache };
}

function backward_for_class(target_class, cache) {
  const L = MODEL.layers;
  let grad_h = new Array(10).fill(0);
  grad_h[target_class] = 1.0;
  
  for (let l = L.length - 1; l >= 0; l--) {
    if (l < L.length - 1) {
      const z = cache[l + 1].z;
      for (let i = 0; i < z.length; i++) {
        if (z[i] <= 0) grad_h[i] = 0;
      }
    }
    const W = L[l].W;
    const grad_in = new Array(W.length).fill(0);
    for (let i = 0; i < W.length; i++) {
      let sum = 0;
      const row = W[i];
      const len = row.length;
      for (let j = 0; j < len; j++) sum += row[j] * grad_h[j];
      grad_in[i] = sum;
    }
    grad_h = grad_in;
  }
  return grad_h; // gradient w.r.t input (784)
}

function train_step(input, target_class, lr = 0.01) {
  const { probs, cache } = forward_with_cache(input);
  let g = probs.slice();
  g[target_class] -= 1; // gradient of cross-entropy loss wrt logits
  
  // Freeze hidden layers to completely prevent catastrophic forgetting.
  // ONLY fine-tune the final classification layer (Linear Probing).
  const L = MODEL.layers;
  const l = L.length - 1; 
  const W = L[l].W;
  const b = L[l].b;
  const h_in = cache[l].h; // previous layer activation
  
  for (let i = 0; i < W.length; i++) {
    const h_val = h_in[i];
    const row = W[i];
    for (let j = 0; j < row.length; j++) {
      row[j] -= lr * h_val * g[j];
    }
  }
  for (let j = 0; j < b.length; j++) {
    b[j] -= lr * g[j];
  }
}

function predict() {
  const input = preprocess();
  if (!input) {
    statusEl.textContent = "Draw something first ✏️";
    document.getElementById("correction").style.display = "none";
    return;
  }
  last_input = input;
  document.getElementById("correction").style.display = "block";
  const { probs, cache } = forward_with_cache(input);
  const best = probs.indexOf(Math.max(...probs));
  digitEl.textContent = best;
  confEl.textContent = `${(probs[best] * 100).toFixed(1)}% confident`;
  for (let d = 0; d <= 9; d++) {
    const row = document.getElementById("bar" + d);
    row.querySelector(".fill").style.width = (probs[d] * 100).toFixed(1) + "%";
    row.querySelector(".pct").textContent = "%" + (probs[d] * 100).toFixed(1);
    row.classList.toggle("top", d === best);
  }
  
  // -- Draw Heatmap (Gradient x Input) --
  const grad = backward_for_class(best, cache);
  let maxVal = 0;
  const saliency = new Array(784).fill(0);
  for (let i = 0; i < 784; i++) {
    saliency[i] = grad[i] * input[i]; // How much this specific stroke helped
    if (Math.abs(saliency[i]) > maxVal) maxVal = Math.abs(saliency[i]);
  }
  
  const heatmap = document.getElementById("heatmap");
  if (heatmap) {
    const hCtx = heatmap.getContext("2d");
    const hImg = hCtx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const idx = i * 4;
      const val = maxVal > 0 ? saliency[i] / maxVal : 0;
      const base = input[i] * 255;
      
      // If it's a stroke (base > 0), color it based on contribution
      if (val > 0) {
        // Positive: tint green
        hImg.data[idx] = Math.max(0, base - val * 255); // R
        hImg.data[idx + 1] = Math.min(255, base + val * 255); // G
        hImg.data[idx + 2] = Math.max(0, base - val * 255); // B
      } else if (val < 0) {
        // Negative: tint red
        hImg.data[idx] = Math.min(255, base - val * 255); // R (-val is positive)
        hImg.data[idx + 1] = Math.max(0, base + val * 255); // G
        hImg.data[idx + 2] = Math.max(0, base + val * 255); // B
      } else {
        // Neutral or background
        hImg.data[idx] = base;
        hImg.data[idx + 1] = base;
        hImg.data[idx + 2] = base;
      }
      hImg.data[idx + 3] = 255; // Opaque
    }
    hCtx.putImageData(hImg, 0, 0);
  }
  
  // Only update status if it wasn't just updated by the correction feedback
  if (!statusEl.textContent.includes("nudged")) {
    statusEl.textContent = "Prediction ready ✅ — keep drawing on top to change it.";
  }
}
