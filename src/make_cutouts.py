"""Stellt Billy auf den drei Fotos frei (Hintergrund, Personen, Gegenstände weg).

Ergebnis: bilder/freigestellt/billy_{A,B,C}.png (RGBA) und cutouts.json.
Das Video nutzt nur diese PNGs; die Modelle werden nur hier gebraucht.

Modelle (über rembg): BiRefNet für die weiche Fellkontur, dazu bei Foto C
Segment Anything (SAM) mit Klickpunkten, um nur den Hund und nicht die
Person dahinter zu erwischen.

Aufruf:  python3 src/make_cutouts.py          (alle drei Fotos)
         python3 src/make_cutouts.py C        (nur Foto C)
"""
import json
import os
import sys

import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'bilder', 'freigestellt')

PHOTOS = {
    'A': dict(file='bilder/d31ad93b38b30e8aafbaaa2372b6c3cc.jpeg', scale=0.75),
    'B': dict(file='bilder/ed453128c9dc6df5e54131ae1af421_livephoto.jpeg', scale=1.0),
    # Foto C: Billy sitzt vor einer Person -> SAM mit positiven (Billy) und
    # negativen Punkten (Person, Bank), Koordinaten im Originalbild.
    'C': dict(file='bilder/IMG_1332.jpeg', scale=1.0,
              sam=[[520, 600, 1], [560, 800, 1], [470, 960, 1], [610, 990, 1], [650, 700, 1],
                   [520, 240, 0], [560, 400, 0], [370, 420, 0], [405, 700, 0], [390, 785, 0],
                   [640, 540, 0], [200, 700, 0]]),
}


def load(path):
    return ImageOps.exif_transpose(Image.open(os.path.join(ROOT, path))).convert('RGB')


def birefnet_mask(im, session):
    from rembg import remove
    small = im.copy()
    small.thumbnail((1600, 1600))
    m = remove(small, session=session, only_mask=True)
    m = np.asarray(m.convert('L')).astype(np.float32) / 255
    return cv2.resize(m, im.size, interpolation=cv2.INTER_CUBIC).clip(0, 1)


def sam_mask(im, points):
    from rembg import new_session, remove
    sc = min(1.0, 1600 / max(im.size))
    small = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)))
    pr = [{"type": "point", "data": [int(x * sc), int(y * sc)], "label": l} for x, y, l in points]
    m = remove(small, session=new_session('sam'), only_mask=True, sam_prompt=pr)
    m = np.asarray(m.convert('L')) > 127
    m = cv2.resize(m.astype(np.uint8), im.size, interpolation=cv2.INTER_NEAREST) > 0
    return m


def largest_component(alpha):
    lab, n = ndimage.label(alpha > 0.5)
    if n <= 1:
        return alpha
    sizes = ndimage.sum(np.ones_like(alpha), lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    keep = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 6)
    return alpha * keep


def decontaminate(img, a, sigma=14):
    """Hintergrundfarbe aus halbtransparenten Fellhaaren herausrechnen."""
    I = img.astype(np.float32)
    inv = 1 - a
    B = cv2.GaussianBlur(I * inv[..., None], (0, 0), sigma) / (cv2.GaussianBlur(inv, (0, 0), sigma)[..., None] + 1e-4)
    Fin = cv2.GaussianBlur(I * a[..., None], (0, 0), sigma) / (cv2.GaussianBlur(a, (0, 0), sigma)[..., None] + 1e-4)
    F = (I - inv[..., None] * B) / np.maximum(a, 0.35)[..., None]
    w = np.clip((a - 0.15) / 0.5, 0, 1)[..., None]
    F = F * w + Fin * (1 - w)
    keep = (a > 0.98)[..., None]
    return np.clip(np.where(keep, I, F), 0, 255).astype(np.uint8)


def main():
    # Jedes Foto in einem eigenen Prozess (BiRefNet braucht viel Arbeitsspeicher)
    if len(sys.argv) == 1:
        import subprocess
        for k in PHOTOS:
            subprocess.run([sys.executable, os.path.abspath(__file__), k], check=True)
        return
    from rembg import new_session
    os.makedirs(OUT, exist_ok=True)
    session = new_session('birefnet-general')
    mpath = os.path.join(OUT, 'cutouts.json')
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for k in sys.argv[1:]:
        spec = PHOTOS[k]
        im = load(spec['file'])
        a = birefnet_mask(im, session)
        if 'sam' in spec:
            s = sam_mask(im, spec['sam'])
            s = ndimage.binary_fill_holes(s)
            s = cv2.dilate(s.astype(np.uint8), np.ones((13, 13), np.uint8))
            a = a * cv2.GaussianBlur(s.astype(np.float32), (0, 0), 3)
        a = largest_component(a)
        img = np.asarray(im)
        rgb = decontaminate(img, a)
        sc = spec['scale']
        if sc != 1.0:
            rgb = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        rgb[a < 0.004] = 0
        rgba = np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)])
        name = f'billy_{k}.png'
        Image.fromarray(rgba).save(os.path.join(OUT, name), optimize=True)
        meta[k] = dict(file=f'bilder/freigestellt/{name}', scale=sc, source=spec['file'],
                       size=list(im.size))
        print('freigestellt:', name, rgba.shape, file=sys.stderr)
    with open(mpath, 'w') as f:
        json.dump(dict(sorted(meta.items())), f, indent=2)


if __name__ == '__main__':
    main()
