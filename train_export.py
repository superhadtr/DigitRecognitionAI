"""Buyuk agi egitir (784->128->64->10) ve agirliklari docs/weights.json'a cikarir."""
import json
import struct
from array import array
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
INPUT = BASE / "data" / "mnist"

ARCH = [784, 128, 64, 10]
EPOCHS, LR, BS = 10, 0.1, 64


def pick(*names):
    for n in names:
        p = INPUT / n
        if p.exists():
            return str(p)
    raise FileNotFoundError(names)


def read_images_labels(images_filepath, labels_filepath):
    with open(labels_filepath, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = array("B", f.read())
    with open(images_filepath, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = array("B", f.read())
    images = np.array(image_data, dtype=np.float64).reshape(size, rows * cols) / 255.0
    return images, np.array(labels, dtype=np.int64)


def forward_cache(x, layers):
    """Ileri besleme; ara degerleri geri yayilim icin saklar."""
    cache = [x]
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.maximum(0, x)
        cache.append(x)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    out = e / e.sum(axis=1, keepdims=True)
    return out, cache


def main():
    np.random.seed(41)
    print("Veri yukleniyor...", flush=True)
    x_train, y_train = read_images_labels(
        pick("train-images-idx3-ubyte/train-images-idx3-ubyte", "train-images.idx3-ubyte"),
        pick("train-labels-idx1-ubyte/train-labels-idx1-ubyte", "train-labels.idx1-ubyte"))
    x_test, y_test = read_images_labels(
        pick("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"),
        pick("t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"))
    print(f"train={x_train.shape} test={x_test.shape} arch={ARCH}", flush=True)

    layers = []
    for in_s, out_s in zip(ARCH[:-1], ARCH[1:]):
        layers.append([np.random.randn(in_s, out_s) * np.sqrt(2.0 / in_s),
                       np.zeros((1, out_s))])

    m = x_train.shape[0]
    for ep in range(EPOCHS):
        idx = np.arange(m)
        np.random.shuffle(idx)
        xs, ys = x_train[idx], y_train[idx]
        tot = 0.0
        for i in range(0, m, BS):
            xb, yb = xs[i:i + BS], ys[i:i + BS]
            p, cache = forward_cache(xb, layers)
            tot += -np.sum(np.log(p[np.arange(len(yb)), yb] + 1e-12)) / len(yb)
            # geri yayilim: softmax+cross-entropy -> relu katmanlari
            g = p.copy()
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            for li in reversed(range(len(layers))):
                W, b = layers[li]
                gw = cache[li].T @ g
                gin = g @ W.T
                if li > 0:
                    gin[cache[li] <= 0] = 0  # relu (Not: cache[li] relu SONRASI deger; <=0 maskesi ayni)
                W -= LR * gw
                b -= LR * g.sum(axis=0, keepdims=True)
                g = gin
        probs, _ = forward_cache(x_train, layers)
        acc = float((probs.argmax(1) == y_train).mean())
        print(f"Epoch {ep+1}/{EPOCHS} loss={tot/(m/BS):.4f} train_acc={acc*100:.2f}%", flush=True)

    probs, _ = forward_cache(x_test, layers)
    pred = probs.argmax(1)
    test_acc = float((pred == y_test).mean())
    print(f"TEST ACCURACY: {test_acc*100:.2f}%", flush=True)
    print("Sinif bazinda:", flush=True)
    for d in range(10):
        mk = y_test == d
        print(f"  {d}: %{(pred[mk] == d).mean()*100:.1f}", flush=True)

    out = {
        "arch": ARCH,
        "test_accuracy": test_acc,
        "layers": [{"W": W.tolist(), "b": b.tolist()[0]} for W, b in layers],
    }
    docs = BASE / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "weights.json").write_text(json.dumps(out))
    print(f"Kaydedildi: docs/weights.json ({(docs/'weights.json').stat().st_size/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
