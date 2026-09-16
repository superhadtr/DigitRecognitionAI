# NumberPredictionAI ✍️🔢

Draw a handwritten digit (0–9) in the browser and a neural network trained on MNIST will guess what it is — with a confidence percentage for every digit.

**🌐 Live demo:** https://superhadtr.github.io/numberPredictionAI/

## How it works

1. You draw on a canvas (black background, white pen — the MNIST format).
2. The drawing is preprocessed in JavaScript: bounding-box crop → fit into 20×20 → centered on a 28×28 grid — the same pipeline used for training data.
3. A feedforward neural network runs **entirely in your browser** (no server):
   - `784 → 10 → 10` fully connected layers, ReLU activations, softmax output
   - Weights exported from the Python training script (`train_export.py`)
   - Test accuracy on MNIST: **93.27%**
4. You see the top prediction plus a percentage bar for each digit 0–9, and a preview of the 28×28 image the network actually sees.

## Project structure

```
docs/
  index.html     # drawing UI (served by GitHub Pages)
  app.js         # preprocessing + forward pass in vanilla JS
  weights.json   # exported trained weights (~170 KB)
train_export.py  # trains the network with NumPy, exports weights.json
neural_network.ipynb  # original training notebook
create_28x28.py  # helper: convert any image to 28×28 grayscale
data/custom/     # example hand-drawn digits
```

## Run locally

No build step — it's static HTML/JS. From this folder:

```bash
python3 -m http.server --directory docs 8000
# open http://localhost:8000
```

To retrain and regenerate the weights:

```bash
python train_export.py   # trains ~10 epochs, writes docs/weights.json
```

## Tech

Python (NumPy, Pillow) for training · Vanilla HTML/CSS/JS for inference · Hosted on GitHub Pages.
