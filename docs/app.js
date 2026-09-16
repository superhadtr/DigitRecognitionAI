// Rakam tanima: cizim -> 28x28 on-isleme -> ileri besleme -> yuzdeler
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

// --- canvas kurulumu: siyah zemin (MNIST standardi) ---
function resetPad() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, N, N);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
resetPad();

// --- cizim ---
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
  ctx.beginPath(); // yarim kalan cizgiyi kapat
  if (MODEL) predict(); // otomatik tahmin
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
});
predictBtn.addEventListener("click", predict);

// --- yuzde cubuklari ---
for (let d = 0; d <= 9; d++) {
  const row = document.createElement("div");
  row.className = "bar";
  row.id = "bar" + d;
  row.innerHTML = `<span class="lbl">${d}</span>
    <div class="track"><div class="fill"></div></div>
    <span class="pct">%0.0</span>`;
  barsEl.appendChild(row);
}

// --- model yukle ---
fetch("weights.json")
  .then((r) => {
    if (!r.ok) throw new Error("weights.json bulunamadi");
    return r.json();
  })
  .then((w) => {
    MODEL = w;
    predictBtn.disabled = false;
    const acc = (w.test_accuracy * 100).toFixed(2);
    statusEl.textContent = `Model ready ✅ (test accuracy: %${acc}) — draw and release, it predicts automatically.`;
  })
  .catch((err) => {
    statusEl.textContent = "Failed to load model: " + err.message;
  });

// --- 28x28 on-isleme (senin Python kodundakiyle ayni mantik) ---
function preprocess() {
  const img = ctx.getImageData(0, 0, N, N).data;
  let x0 = N, y0 = N, x1 = -1, y1 = -1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (img[(y * N + x) * 4] > 20) { // beyaz piksel var mi
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null; // bos canvas

  const m = 14;
  x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m);
  x1 = Math.min(N - 1, x1 + m); y1 = Math.min(N - 1, y1 + m);
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1;

  // kirpilmis bolgeyi gecici canvas'a koy
  const crop = document.createElement("canvas");
  crop.width = cw; crop.height = ch;
  crop.getContext("2d").putImageData(ctx.getImageData(x0, y0, cw, ch), 0, 0);

  // en-boy oranini koruyup 20x20'ye sigdir, 28x28'de ortala
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

// --- ileri besleme: x -> relu(x*W1+b1) -> relu(h*W2+b2) -> softmax ---
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
function forward(input) {
  const h1 = relu(matvec(input, MODEL.W1, MODEL.b1));
  const h2 = relu(matvec(h1, MODEL.W2, MODEL.b2));
  const mx = Math.max(...h2);
  const e = h2.map((v) => Math.exp(v - mx));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

function predict() {
  const input = preprocess();
  if (!input) {
    statusEl.textContent = "Draw something first ✏️";
    return;
  }
  const probs = forward(input);
  const best = probs.indexOf(Math.max(...probs));
  digitEl.textContent = best;
  confEl.textContent = `${(probs[best] * 100).toFixed(1)}% confident`;
  for (let d = 0; d <= 9; d++) {
    const row = document.getElementById("bar" + d);
    row.querySelector(".fill").style.width = (probs[d] * 100).toFixed(1) + "%";
    row.querySelector(".pct").textContent = "%" + (probs[d] * 100).toFixed(1);
    row.classList.toggle("top", d === best);
  }
  statusEl.textContent = "Prediction ready ✅ — keep drawing on top to change it.";
}
