"""Notebook'taki agi egitir ve tarayici icin agirliklari site/docs/weights.json'a cikarir."""
import json
import struct
from array import array
from os.path import join
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
INPUT = BASE / "data" / "mnist"

# MNIST dosya yollarini bul (hem duz hem klasor ici duruyor)
def pick(*names):
    for n in names:
        p = INPUT / n
        if p.exists():
            return str(p)
    raise FileNotFoundError(names)

training_images_filepath = pick("train-images-idx3-ubyte/train-images-idx3-ubyte", "train-images.idx3-ubyte")
training_labels_filepath = pick("train-labels-idx1-ubyte/train-labels-idx1-ubyte", "train-labels.idx1-ubyte")
test_images_filepath = pick("t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte")
test_labels_filepath = pick("t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte")


def read_images_labels(images_filepath, labels_filepath):
    with open(labels_filepath, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = array("B", f.read())
    with open(images_filepath, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = array("B", f.read())
    images = np.array(image_data, dtype=np.float64).reshape(size, rows * cols) / 255.0
    return images, np.array(labels, dtype=np.int64)


class LinearLayer:
    def __init__(self, in_s, out_s):
        self.weights = np.random.randn(in_s, out_s) * np.sqrt(2.0 / in_s)
        self.biases = np.zeros((1, out_s))

    def forward(self, x):
        self.inp = x
        return x @ self.weights + self.biases

    def backward(self, grad, lr):
        gw = self.inp.T @ grad
        gin = grad @ self.weights.T
        self.weights -= lr * gw
        self.biases -= lr * np.sum(grad, axis=0, keepdims=True)
        return gin


class ReLU:
    def forward(self, x):
        self.inp = x
        return np.maximum(0, x)

    def backward(self, grad, lr):
        g = grad.copy()
        g[self.inp <= 0] = 0
        return g


class Softmax:
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = e / np.sum(e, axis=1, keepdims=True)
        return self.out

    def backward(self, grad, lr):
        return grad


class CrossEntropy:
    def forward(self, p, t):
        self.p, self.t = p, t
        m = t.shape[0]
        return -np.sum(np.log(p[np.arange(m), t] + 1e-12)) / m

    def backward(self):
        m = self.t.shape[0]
        g = self.p.copy()
        g[np.arange(m), self.t] -= 1
        return g / m


def main():
    np.random.seed(41)
    print("Veri yukleniyor...", flush=True)
    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)
    print(f"train={x_train.shape} test={x_test.shape}", flush=True)

    W1, A1, W2, A2, SM, LOSS = LinearLayer(784, 10), ReLU(), LinearLayer(10, 10), ReLU(), Softmax(), CrossEntropy()
    lr, epochs, bs = 0.1, 10, 64
    m = x_train.shape[0]

    def forward(x):
        o = W1.forward(x)
        o = A1.forward(o)
        o = W2.forward(o)
        o = A2.forward(o)
        return SM.forward(o)

    for ep in range(epochs):
        idx = np.arange(m)
        np.random.shuffle(idx)
        xs, ys = x_train[idx], y_train[idx]
        tot = 0.0
        for i in range(0, m, bs):
            xb, yb = xs[i:i + bs], ys[i:i + bs]
            p = forward(xb)
            tot += LOSS.forward(p, yb)
            g = LOSS.backward()
            g = SM.backward(g, lr)
            g = A2.backward(g, lr)
            g = W2.backward(g, lr)
            g = A1.backward(g, lr)
            W1.backward(g, lr)
        acc = np.mean(np.argmax(forward(x_train), axis=1) == y_train)
        print(f"Epoch {ep+1}/{epochs} loss={tot/(m/bs):.4f} train_acc={acc*100:.2f}%", flush=True)

    test_acc = float(np.mean(np.argmax(forward(x_test), axis=1) == y_test))
    print(f"TEST ACCURACY: {test_acc*100:.2f}%", flush=True)

    out = {
        "arch": [784, 10, 10],
        "test_accuracy": test_acc,
        "W1": W1.weights.tolist(),
        "b1": W1.biases.tolist()[0],
        "W2": W2.weights.tolist(),
        "b2": W2.biases.tolist()[0],
    }
    docs = BASE / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "weights.json").write_text(json.dumps(out))
    print(f"Kaydedildi: docs/weights.json ({(docs/'weights.json').stat().st_size/1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
