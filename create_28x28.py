"""
28x28 Grayscale goruntu olusturma + MNIST formatina cevirme
Kullanim:
  python create_28x28.py --ornek           # MNIST'ten ornek bir rakami PNG olarak kaydet
  python create_28x28.py --input foto.png --label 5 --output data/custom/benim_5.png
  python create_28x28.py --ciz 3           # kod ile basit bir rakam ciz (ornek)
"""
import argparse
import struct
from array import array
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).parent
MNIST_DIR = BASE / "data" / "mnist"
CUSTOM_DIR = BASE / "data" / "custom"
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

def mnist_oku(images_path, labels_path, index=0):
    with open(labels_path, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = array("B", f.read())
    with open(images_path, 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        img_data = array("B", f.read())
    img = np.array(img_data[index*rows*cols:(index+1)*rows*cols], dtype=np.uint8).reshape(rows, cols)
    return img, int(labels[index])

def ornek_kaydet(index=0):
    train_img = MNIST_DIR / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"
    train_lbl = MNIST_DIR / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte"
    # duz dosya olarak da duruyor, hangisi varsa onu kullan
    if not train_img.exists():
        train_img = MNIST_DIR / "train-images.idx3-ubyte"
    if not train_lbl.exists():
        train_lbl = MNIST_DIR / "train-labels.idx1-ubyte"
    img, label = mnist_oku(str(train_img), str(train_lbl), index)
    out = CUSTOM_DIR / f"mnist_ornek_{index}_label{label}.png"
    Image.fromarray(img, mode="L").save(out)
    print(f"Kaydedildi: {out} | label={label} | shape={img.shape} | mod=grayscale(L)")
    # ayni goruntuyu npy olarak da kaydet (egitime data eklemek icin en kolayi)
    np.save(str(out.with_suffix(".npy")), img)
    print(f"Kaydedildi: {out.with_suffix('.npy')} (numpy array, shape {img.shape})")
    return out

def herhangi_resmi_cevir(input_path, output_path, label=None):
    """Herhangi bir JPG/PNG cizimi 28x28 grayscale'e cevir (MNIST standardi)."""
    img = Image.open(input_path).convert("L")  # grayscale
    # MNIST: siyah zemin + beyaz rakam. Eger beyaz zeminli cizdiysen ters cevir:
    # img = Image.fromarray(255 - np.array(img))
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output_path)
    np.save(str(output_path.with_suffix(".npy")), arr)
    print(f"Cevirildi: {input_path} -> {output_path} | shape={arr.shape}")
    if label is not None:
        print(f"Etiketin: {label} (egitime eklerken kullan)")
    return output_path

def kod_ile_ciz(rakam=3):
    """Hic resmin yoksa: PIL ile 28x28'e basit bir rakam ciz."""
    img = Image.new("L", (28, 28), color=0)  # siyah zemin (MNIST gibi)
    d = ImageDraw.Draw(img)
    try:
        # sistem fontu bulmaya calis, olmazsa default
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()
    d.text((5, 1), str(rakam), fill=255, font=font)
    out = CUSTOM_DIR / f"el_yapimi_{rakam}.png"
    img.save(out)
    np.save(str(out.with_suffix(".npy")), np.array(img))
    print(f"Cizildi: {out}")
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ornek", action="store_true", help="MNIST'ten ornek kaydet")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--input", type=str, help="cevrilecek resim yolu")
    p.add_argument("--output", type=str, help="cikti PNG yolu")
    p.add_argument("--label", type=int, default=None)
    p.add_argument("--ciz", type=int, default=None, help="ornek: --ciz 3")
    a = p.parse_args()

    if a.ornek or (not a.input and a.ciz is None):
        ornek_kaydet(a.index)
    if a.ciz is not None:
        kod_ile_ciz(a.ciz)
    if a.input:
        out = a.output or str(CUSTOM_DIR / "benim_cizimim.png")
        herhangi_resmi_cevir(a.input, out, a.label)
