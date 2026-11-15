"""Rendert das Musikvideo "Lord Billy – Oye Como Wau" im Musik-TV-Stil (1080x1920, 30 fps).

Aus den drei Fotos entstehen Kamerafahrten mit harten Schnitten im Takt.
Billys Unterkiefer bewegt sich synchron zu den gesungenen Phonemen, vor dem
Maul steht ein Mikrofon auf einem Ständer. Dazu Sender-Logo, Credits unten
links, Bühnenlicht, RGB-Versatz, Stroboskop, Split-Screen im Refrain, große
Wörter im Takt und Karaoke-Untertitel der gerade gesungenen Zeile.

Aufruf:
  python3 src/make_video.py build/song.wav video/LordBilly_OyeComoWau.mp4
  python3 src/make_video.py --preview 13.2,22.0 out_dir     # Einzelbilder
"""
import os
import sys
import subprocess
from multiprocessing import Pool

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

import timeline as TL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1080, 1920
FPS = TL.FPS
NF = int(round(TL.DURATION * FPS))
O = np.array([W / 2, H / 2])

FONT_DIR = '/usr/share/fonts/opentype/inter/'
F_BLACK = FONT_DIR + 'InterDisplay-Black.otf'
F_XBOLD = FONT_DIR + 'InterDisplay-ExtraBold.otf'
F_BOLD = FONT_DIR + 'InterDisplay-Bold.otf'
F_SEMI = FONT_DIR + 'InterDisplay-SemiBold.otf'
F_ITAL = FONT_DIR + 'InterDisplay-SemiBoldItalic.otf'
F_SYM = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

GOLD = (255, 200, 61)
ORANGE = (255, 128, 40)
PINK = (255, 60, 150)
CYAN = (40, 220, 255)
WHITE = (255, 255, 255)

# Maul-Geometrie je Foto (Pixel im Originalbild) und Mikrofon-Position
# (Mitte des Mikrofonkorbs, Radius, Neigung des Griffs in Grad).
IMAGES = {
    'A': dict(file='bilder/d31ad93b38b30e8aafbaaa2372b6c3cc.jpeg', prescale=0.75,
              mouth=dict(x=1470, lip=1345, gap=70, jaw=1700, decay=420, half=330, slope=0.0, amax=115),
              mic=dict(x=1745, y=1745, r=150, tilt=30)),
    'B': dict(file='bilder/ed453128c9dc6df5e54131ae1af421_livephoto.jpeg', prescale=1.0,
              mouth=dict(x=965, lip=842, gap=42, jaw=975, decay=230, half=180, slope=0.06, amax=58),
              mic=dict(x=1160, y=1035, r=80, tilt=32)),
    'C': dict(file='bilder/IMG_1332.jpeg', prescale=1.0, sharpen=True,
              mouth=dict(x=521, lip=627, gap=9, jaw=683, decay=95, half=52, slope=0.0, amax=19),
              mic=dict(x=592, y=716, r=27, tilt=30)),
}

# Einstellungen. Kamera = (Mitte x, Mitte y, Quellpixel je Ausgabepixel, Drehung in Grad)
# Split-Screen: (Foto, Kamera Anfang, Kamera Ende, Gesichtsmitte y im Foto)
SPLIT_A = ('A', (1460, 1700, 1.85, 0), (1460, 1700, 1.7, 0), 1230)
SPLIT_B = ('B', (930, 960, 1.0, 0), (930, 960, 0.92, 0), 720)
SPLIT_C = ('C', (528, 600, 0.47, 0), (528, 600, 0.44, 0), 600)
SHOTS = [
    dict(t0=0.0, t1=4.0, img='C', c0=(540, 545, 0.54, 0), c1=(528, 600, 0.47, 0), flash=False),
    dict(t0=4.0, t1=6.0, img='B', c0=(960, 1280, 1.30, 0), c1=(940, 1150, 1.12, 0), flash=False),
    dict(t0=6.0, t1=8.0, img='A', c0=(1500, 2016, 2.0, 0), c1=(1480, 1900, 1.8, 0), flash=True),
    dict(t0=8.0, t1=12.0, img='B', c0=(930, 960, 1.0, 0), c1=(915, 880, 0.85, 0), flash=True),
    dict(t0=12.0, t1=16.0, img='A', c0=(1470, 1700, 1.65, 0), c1=(1460, 1450, 1.3, 0), flash=True),
    dict(t0=16.0, t1=20.0, img='B', c0=(930, 960, 1.0, 0), c1=(950, 860, 0.8, 0), flash=True),
    dict(t0=20.0, t1=24.0, img='A', c0=(1460, 1500, 1.3, -3), c1=(1460, 1420, 1.1, 2), flash=True),
    dict(t0=24.0, t1=28.0, img='B', c0=(900, 1000, 1.1, 3), c1=(950, 860, 0.85, -2), flash=True),
    dict(t0=28.0, t1=30.0, img='A', c0=(1460, 1450, 1.3, 0), c1=(1460, 1430, 1.15, 0), flash=True),
    dict(t0=30.0, t1=32.0, split=[SPLIT_A, SPLIT_B, SPLIT_C], flash=True),
    dict(t0=32.0, t1=34.0, img='B', c0=(950, 870, 0.85, 0), c1=(940, 870, 0.75, 0), flash=True),
    dict(t0=34.0, t1=36.0, img='A', c0=(1460, 1450, 1.2, 0), c1=(1460, 1400, 1.05, -2.0), flash=True),
    dict(t0=36.0, t1=37.0, img='C', c0=(528, 600, 0.48, 0), c1=(528, 600, 0.44, 0), flash=True, tint=0),
    dict(t0=37.0, t1=38.0, img='B', c0=(920, 820, 0.95, 0), c1=(920, 820, 0.85, 0), flash=True, tint=1),
    dict(t0=38.0, t1=39.0, split=[SPLIT_B, SPLIT_C, SPLIT_A], flash=True, tint=2),
    dict(t0=39.0, t1=40.0, img='A', c0=(1460, 1450, 1.15, 0), c1=(1460, 1420, 1.0, -2.0), flash=True, tint=0),
    dict(t0=40.0, t1=41.0, img='B', c0=(950, 870, 0.8, 0), c1=(950, 870, 0.7, 0), flash=True, tint=1),
    dict(t0=41.0, t1=42.0, img='C', c0=(540, 545, 0.54, 0), c1=(530, 560, 0.5, 0), flash=True, tint=2),
    dict(t0=42.0, t1=45.0, img='A', c0=(1460, 1430, 1.1, -1.5), c1=(1480, 1920, 2.0, 0), flash=True),
]
CROSSFADES = {4.0: 0.4}
TINTS = [((1.0, 0.35, 0.75), (0.15, 0.0, 0.35)),     # pink
         ((0.35, 0.9, 1.0), (0.0, 0.1, 0.35)),       # cyan
         ((1.0, 0.8, 0.25), (0.3, 0.05, 0.0))]       # gold
CHORUS_TONE = ((1.0, 0.62, 0.82), (0.05, 0.12, 0.42))


# ----------------------------------------------------------------- Hilfen
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(x):
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


KICKS = TL.kick_times()
BEATS = np.array([TL.bt(b, k) for b in range(2, 21) for k in range(4)] + [TL.bt(21)])


def env_after(times, t, tau):
    past = times[times <= t]
    if len(past) == 0:
        return 0.0
    return float(np.exp(-(t - past[-1]) / tau))


def kick_env(t):
    return env_after(KICKS, t, 0.13)


def font(path, size):
    return ImageFont.truetype(path, size)


def text_sprite(text, fnt, fill, stroke=6, stroke_fill=(30, 10, 0), shadow=10):
    """RGBA-Array mit Text, Kontur und weichem Schatten."""
    l, t, r, b = fnt.getbbox(text, stroke_width=stroke)
    pad = shadow * 3 + 4
    w, h = r - l + 2 * pad, b - t + 2 * pad
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    if shadow:
        sh = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l + 3, pad - t + 6), text, font=fnt, fill=(0, 0, 0, 170),
                                stroke_width=stroke, stroke_fill=(0, 0, 0, 170))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(shadow)))
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=fnt, fill=fill + (255,),
                             stroke_width=stroke, stroke_fill=stroke_fill + (255,))
    return np.array(img)


def blend(dst, spr, cx, cy, alpha=1.0):
    """Sprite (RGBA, uint8) mittig bei (cx, cy) auf dst (float32 0..1) legen."""
    if alpha <= 0.003:
        return
    h, w = spr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32) / 255.0
    a = s[..., 3:4] * alpha
    d = dst[ya:yb, xa:xb]
    d *= (1 - a)
    d += s[..., :3] * a


def xform(spr, scale=1.0, angle=0.0):
    """Sprite skalieren und drehen (Leinwand wird passend vergrößert)."""
    if abs(scale - 1) < 0.01 and abs(angle) < 0.1:
        return spr
    h, w = spr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(w * c + h * s) + 2, int(w * s + h * c) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(spr, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


# ----------------------------------------------------------------- Bilder
_IMG = {}


def load_images():
    for k, spec in IMAGES.items():
        im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, spec['file']))).convert('RGB')
        if spec.get('sharpen'):
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
        a = np.asarray(im).astype(np.float32)
        # knackiger Musikvideo-Look: mehr Kontrast, warme Lichter
        a[..., 0] *= 1.05
        a[..., 2] *= 0.94
        a = 255 * (0.5 + 1.14 * (a / 255 - 0.5))
        a = np.clip(a, 0, 255).astype(np.uint8)
        f = spec['prescale']
        if f != 1.0:
            a = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        _IMG[k] = dict(img=a, f=f, size=(im.size[0], im.size[1]))


def clamp_cam(k, cx, cy, s, rot):
    iw, ih = _IMG[k]['size']
    th = np.radians(rot)
    hw = (W / 2 * abs(np.cos(th)) + H / 2 * abs(np.sin(th))) * s
    hh = (W / 2 * abs(np.sin(th)) + H / 2 * abs(np.cos(th))) * s
    cx = iw / 2 if 2 * hw >= iw else np.clip(cx, hw, iw - hw)
    cy = ih / 2 if 2 * hh >= ih else np.clip(cy, hh, ih - hh)
    return cx, cy, s, rot


def camera_for(k, c0, c1, t0, t1, t):
    u = ease((t - t0) / (t1 - t0))
    cx, cy, s, rot = [a + (b - a) * u for a, b in zip(c0, c1)]
    sec = TL.section_at(t)
    hot = sec in ('chorus', 'solo')
    ke = kick_env(t)
    s *= 1 - (0.035 if hot else 0.018) * ke
    rot += (1.4 if sec == 'solo' else 0.8) * np.sin(2 * np.pi * t / (2 * TL.BAR))
    if hot:   # Kamerawackler auf dem Kick
        rng = np.random.default_rng(int(t * 1000))
        cx += s * 9 * ke * rng.uniform(-1, 1)
        cy += s * 9 * ke * rng.uniform(-1, 1)
    return clamp_cam(k, cx, cy, s, rot)


def camera(sh, t):
    return camera_for(sh['img'], sh['c0'], sh['c1'], sh['t0'], sh['t1'], t)


def render_photo(k, cam):
    cx, cy, s, rot = cam
    d = _IMG[k]
    f = d['f']
    th = np.radians(rot)
    c, sn = np.cos(th), np.sin(th)
    # Ausgabe p -> Quelle q = C + s R (p - O), danach Vorskalierung f
    m = np.array([[s * c, -s * sn, 0], [s * sn, s * c, 0]], dtype=np.float64)
    m[:, 2] = np.array([cx, cy]) - m[:, :2] @ O
    m *= f
    return cv2.warpAffine(d['img'], m, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT101)


def to_out(cam, q):
    cx, cy, s, rot = cam
    th = np.radians(rot)
    c, sn = np.cos(th), np.sin(th)
    v = (np.asarray(q, dtype=float) - np.array([cx, cy])) / s
    return O + np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1]])


def mouth_geom(k, cam):
    m = IMAGES[k]['mouth']
    s = cam[2]
    xc, lip = to_out(cam, (m['x'], m['lip']))
    slope = np.tan(np.arctan(m['slope']) - np.radians(cam[3]))
    return dict(xc=xc, lip=lip, gap=m['gap'] / s, J=(m['jaw'] - m['lip']) / s, D=m['decay'] / s,
                half=m['half'] / s, slope=slope, amax=m['amax'] / s)


def open_mouth(frame, g, amount):
    """Öffnet den Unterkiefer: Bereich unter der Oberlippe wird nach unten gedehnt."""
    A = amount * g['amax']
    if A < 0.4:
        return frame
    hw = 1.3 * g['half']
    x0, x1 = int(max(g['xc'] - hw, 0)), int(min(g['xc'] + hw, W - 1))
    span = g['J'] + g['D'] + 4
    offs = g['slope'] * (np.arange(x0, x1 + 1) - g['xc'])
    y0 = int(max(g['lip'] + offs.min() - 6, 0))
    y1 = int(min(g['lip'] + offs.max() + span + 6, H - 1))
    if x1 <= x0 or y1 <= y0:
        return frame
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    spread = offs.max() - offs.min()
    ugrid = np.arange(-spread - 12, span + spread + 12, 0.5)
    hgrid = np.where(ugrid < 0, 0,
                     np.where(ugrid < g['gap'], smooth(ugrid / g['gap']),
                              np.where(ugrid < g['J'], 1.0, 1 - smooth((ugrid - g['J']) / g['D']))))
    map_y = np.empty((len(ys), x1 - x0 + 1), np.float32)
    for j, x in enumerate(range(x0, x1 + 1)):
        u = abs(x - g['xc']) / hw
        a = A * (1 - u * u) ** 2 if u < 1 else 0.0
        lipy = g['lip'] + offs[j]
        if a < 0.05:
            map_y[:, j] = ys
            continue
        fwd = ugrid + a * hgrid
        map_y[:, j] = lipy + np.interp(ys - lipy, fwd, ugrid)
    map_x = np.broadcast_to(np.arange(x0, x1 + 1, dtype=np.float32), map_y.shape).copy()
    patch = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    frame = frame.copy()
    frame[y0:y1 + 1, x0:x1 + 1] = patch
    return frame


# ----------------------------------------------------------------- Mikrofon
MIC_R = 170   # Korbradius im Sprite


def build_mic():
    """Gerendertes Gesangsmikrofon (Korb oben, Griff nach unten) als RGBA-Array."""
    R = MIC_R
    Wd, Hd = int(2.4 * R), int(5.6 * R)
    cx, cy = Wd / 2, 1.1 * R
    yy, xx = np.mgrid[0:Hd, 0:Wd].astype(np.float32)
    rgb = np.zeros((Hd, Wd, 3), np.float32)
    alpha = np.zeros((Hd, Wd), np.float32)
    L = np.array([-0.45, -0.55, 0.7])
    L /= np.linalg.norm(L)
    # Griff: konisch, schwarz, mit Glanzstreifen und Chromringen
    y0, y1 = cy + 0.45 * R, cy + 4.3 * R
    u = (yy - y0) / (y1 - y0)
    hw = 0.46 * R * (1 - u) + 0.27 * R * u
    nx = np.clip((xx - cx) / np.maximum(hw, 1), -1, 1)
    nz = np.sqrt(1 - nx ** 2)
    dif = np.clip(nx * L[0] + nz * L[2], 0, 1)
    spec = np.clip(nx * -0.35 + nz * 0.94, 0, 1) ** 40
    body = np.array([0.07, 0.07, 0.08]) + dif[..., None] * 0.16 + spec[..., None] * 0.7
    chrome = np.array([0.55, 0.56, 0.6]) * (0.45 + 0.6 * dif[..., None]) + spec[..., None] * 0.9 \
        + 0.12 * np.sin(nx * 9)[..., None]
    is_ring = ((u >= 0) & (u < 0.1)) | ((u > 0.9) & (u <= 1))
    col = np.where(is_ring[..., None], chrome, body)
    a_h = np.clip(hw - np.abs(xx - cx) + 0.5, 0, 1) * ((u >= 0) & (u <= 1))
    rgb = rgb * (1 - a_h[..., None]) + col * a_h[..., None]
    alpha = np.maximum(alpha, a_h)
    # Halteklammer am Ende
    kx, ky, kr = cx, y1 + 0.18 * R, 0.24 * R
    d = np.sqrt((xx - kx) ** 2 + (yy - ky) ** 2)
    a_k = np.clip(kr - d + 0.5, 0, 1)
    nzk = np.sqrt(np.clip(1 - (d / kr) ** 2, 0, 1))
    dk = np.clip(((xx - kx) / kr) * L[0] + ((yy - ky) / kr) * L[1] + nzk * L[2], 0, 1)
    ck = np.array([0.1, 0.1, 0.11]) + dk[..., None] * 0.35
    rgb = rgb * (1 - a_k[..., None]) + ck * a_k[..., None]
    alpha = np.maximum(alpha, a_k)
    # Korb: Kugel mit Drahtgitter
    dx, dy = (xx - cx) / R, (yy - cy) / R
    r2 = dx ** 2 + dy ** 2
    inside = r2 <= 1.0
    nzs = np.sqrt(np.clip(1 - r2, 0, 1))
    dif = np.clip(dx * L[0] + dy * L[1] + nzs * L[2], 0, 1)
    refl = 2 * dif * nzs - L[2]
    spec = np.clip(refl, 0, 1) ** 25
    lat = np.arcsin(np.clip(dy, -1, 1))
    lon = np.arctan2(dx, np.maximum(nzs, 1e-3))
    k = 15
    g1 = np.abs(np.sin(k * (lat + lon)))
    g2 = np.abs(np.sin(k * (lat - lon)))
    wire = smooth((0.35 - np.minimum(g1, g2)) / 0.2)
    hole = np.array([0.1, 0.1, 0.11]) + 0.12 * dif[..., None]
    metal = np.array([0.62, 0.63, 0.66]) * (0.35 + 0.75 * dif[..., None]) + spec[..., None]
    gcol = hole * (1 - wire[..., None]) + metal * wire[..., None]
    gcol += spec[..., None] * 0.35
    gcol *= (0.55 + 0.45 * nzs[..., None] ** 0.5)
    # Ring zwischen Korb und Griff
    band = (dy > 0.55) & (dy < 0.75)
    gcol = np.where(band[..., None], np.array([0.7, 0.71, 0.74]) * (0.4 + 0.7 * dif[..., None]) + spec[..., None], gcol)
    a_s = np.clip((1 - np.sqrt(r2)) * R + 0.5, 0, 1) * inside
    rgb = rgb * (1 - a_s[..., None]) + gcol * a_s[..., None]
    alpha = np.maximum(alpha, a_s)
    spr = np.dstack([np.clip(rgb, 0, 1) * 255, alpha * 255]).astype(np.uint8)
    return spr, (cx, cy), (kx, ky)


def draw_mic(frame, k, cam, t):
    """Mikrofon samt Ständer passend zur Kamera vor das Maul setzen (uint8-Bild)."""
    spec = IMAGES[k]['mic']
    spr, (cx, cy), (kx, ky) = SPR['mic']
    gx, gy = to_out(cam, (spec['x'], spec['y']))
    r_out = spec['r'] / cam[2]
    sc = r_out / MIC_R
    ang = spec['tilt'] - cam[3] + 1.5 * np.sin(2 * np.pi * t / 3.1)
    m = cv2.getRotationMatrix2D((cx, cy), ang, sc)
    m[0, 2] += gx - cx
    m[1, 2] += gy - cy
    jx, jy = m @ np.array([kx, ky, 1.0])
    # Bounding-Box
    h, w = spr.shape[:2]
    pts = m @ np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    x0, x1 = int(max(pts[0].min() - 4, 0)), int(min(pts[0].max() + 4, W))
    y0, y1 = int(max(min(pts[1].min(), jy) - 4, 0)), H
    if x1 <= x0 or y1 <= y0:
        return frame
    m2 = m.copy()
    m2[0, 2] -= x0
    m2[1, 2] -= y0
    layer = cv2.warpAffine(spr, m2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    # Ständer: senkrechte Chromstange von der Klammer bis zum Bildrand
    pw = max(3, int(0.17 * r_out))
    px0 = int(jx - pw / 2) - x0
    if 0 <= px0 and px0 + pw <= layer.shape[1] and jy < H:
        prof = np.linspace(-1, 1, pw)
        shade = (0.35 + 0.55 * np.clip(1 - (prof + 0.35) ** 2, 0, 1) + 0.4 * np.exp(-((prof + 0.3) / 0.15) ** 2))
        col = np.clip(np.outer(shade, [0.75, 0.76, 0.8]) * 255, 0, 255)
        ys = int(max(jy - y0, 0))
        pole = layer[ys:, px0:px0 + pw]
        under = pole[..., 3:4].astype(np.float32) / 255
        pole[..., :3] = (pole[..., :3] * under + col[None] * (1 - under)).astype(np.uint8)
        pole[..., 3] = 255
    a = layer[..., 3:4].astype(np.float32) / 255
    out = frame.copy()
    reg = out[y0:y1, x0:x1].astype(np.float32)
    # weicher Schlagschatten auf Billy
    sh = cv2.GaussianBlur(a[..., 0], (0, 0), max(2, r_out * 0.15))
    off = int(r_out * 0.25)
    shm = np.zeros_like(sh)
    shm[off:, off:] = sh[:-off or None, :-off or None] if off > 0 else sh
    reg *= (1 - 0.38 * shm[..., None])
    reg = reg * (1 - a) + layer[..., :3].astype(np.float32) * a
    out[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------- Einstellungen rendern
def shot_at(t):
    for sh in SHOTS:
        if sh['t0'] <= t < sh['t1']:
            return sh
    return SHOTS[-1]


def render_single(k, cam, t, mouth):
    fr = render_photo(k, cam)
    fr = open_mouth(fr, mouth_geom(k, cam), mouth)
    return draw_mic(fr, k, cam, t)


def render_shot(sh, t, mouth):
    if 'split' not in sh:
        return render_single(sh['img'], camera(sh, t), t, mouth)
    out = np.zeros((H, W, 3), np.uint8)
    bh = H // 3
    for i, (k, c0, c1, fy) in enumerate(sh['split']):
        cam = camera_for(k, c0, c1, sh['t0'], sh['t1'], t)
        fr = render_single(k, cam, t, mouth)
        fx = IMAGES[k]['mouth']['x']
        cy = int(np.clip(to_out(cam, (fx, fy))[1], bh / 2, H - bh / 2))
        out[i * bh:(i + 1) * bh] = fr[cy - bh // 2:cy - bh // 2 + bh]
    for i in (1, 2):   # Trennlinien
        out[i * bh - 4:i * bh + 4] = (255, 60, 150) if i == 1 else (40, 220, 255)
    return out


def photo_frame(t, mouth):
    for cut, dur in CROSSFADES.items():
        if cut - dur / 2 <= t < cut + dur / 2:
            prev, nxt = shot_at(cut - 1e-3), shot_at(cut)
            w = smooth((t - (cut - dur / 2)) / dur)
            fa = render_shot(prev, min(t, prev['t1'] - 1e-3), mouth)
            fb = render_shot(nxt, max(t, nxt['t0']), mouth)
            return (fa.astype(np.float32) * (1 - w) + fb.astype(np.float32) * w).astype(np.uint8)
    return render_shot(shot_at(t), t, mouth)


# ----------------------------------------------------------------- Licht & Look
LW, LH = 135, 240
_yy, _xx = np.mgrid[0:LH, 0:LW].astype(np.float32)
_xx = (_xx + 0.5) / LW
_yy = (_yy + 0.5) / LH


def beam_colors(t):
    sec = TL.section_at(t)
    if sec == 'chorus':
        return [PINK, CYAN, PINK, CYAN], 0.22
    if sec == 'solo':
        i = int(t) % 3
        cols = [PINK, CYAN, GOLD]
        return [cols[i], cols[(i + 1) % 3], cols[i], cols[(i + 2) % 3]], 0.28
    if sec in ('verse', 'riff'):
        return [GOLD, ORANGE, GOLD, ORANGE], 0.12
    return [GOLD, ORANGE, GOLD, ORANGE], 0.08


def light_maps(t):
    k = kick_env(t)
    r = np.sqrt(((_xx - 0.5) * 1.1) ** 2 + ((_yy - 0.47) * 0.75) ** 2)
    vig = 1 - (0.6 + 0.12 * k) * smooth((r - 0.2) / 0.42)
    bottom = 1 - 0.5 * smooth((_yy - 0.7) / 0.28)       # Lesbarkeit der Untertitel
    mult = vig * bottom
    add = np.zeros((LH, LW, 3), np.float32)
    # Bühnen-Lichtkegel von oben
    cols, inten = beam_colors(t)
    aspect = H / W
    for i, (bx, col) in enumerate(zip((0.08, 0.36, 0.64, 0.92), cols)):
        ang = np.radians(18 * np.sin(2 * np.pi * t / (2 * TL.BAR) + i * 1.7) + (12 if bx < 0.5 else -12))
        dx = _xx - bx
        dy = (_yy + 0.05) * aspect
        a = np.arctan2(dx, dy) - ang
        cone = np.exp(-(a / 0.09) ** 2) * np.exp(-dy / 2.2) * (dy > 0)
        add += cone[..., None] * (np.array(col, np.float32) / 255) * inten * (0.8 + 0.4 * k)
    # weiche Farbflecken (Lens-Flare-Gefühl)
    for col, bx, by, sp, ph in (((1.0, 0.55, 0.15), 0.25, 0.10, 0.11, 0.0),
                                ((1.0, 0.25, 0.5), 0.8, 0.18, 0.07, 2.0)):
        x = bx + 0.25 * np.sin(sp * 2 * np.pi * t + ph)
        y = by + 0.08 * np.cos(sp * 1.7 * np.pi * t + ph)
        d2 = (_xx - x) ** 2 + ((_yy - y) * 0.56) ** 2
        add += np.exp(-d2 / 0.03)[..., None] * np.array(col, np.float32) * 0.12
    mult = cv2.resize(mult, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    add = cv2.resize(add, (W, H), interpolation=cv2.INTER_LINEAR)
    return mult, add


def tone(F, hi, lo, amt):
    """Split-Toning: Lichter Richtung `hi`, Schatten Richtung `lo`."""
    if amt <= 0:
        return F
    l = F.mean(axis=2, keepdims=True)
    toned = l * np.array(hi, np.float32) * 1.15 + (1 - l) * np.array(lo, np.float32) * 0.5
    toned = np.clip(toned * (0.6 + 0.6 * l), 0, 1)
    return F * (1 - amt) + toned * amt


GRAIN = []


def build_grain():
    rng = np.random.default_rng(3)
    for _ in range(6):
        GRAIN.append(cv2.GaussianBlur((rng.standard_normal((H // 2, W // 2)) * 0.022).astype(np.float32), (0, 0), 0.8))


# ----------------------------------------------------------------- Texte
SPR = {}
LYR = []


def build_lyrics():
    fnt = font(F_XBOLD, 56)
    notes = TL.vocal_notes()
    spans = TL.line_spans()
    for li, line in enumerate(TL.LINES):
        ns = [n for n in notes if n['line'] == li]
        chunks = [n['text'] for n in ns]
        widths = [fnt.getlength(c) for c in chunks]
        rows, cur, curw = [], [], 0.0
        word, wordw = [], 0.0
        for i, c in enumerate(chunks):
            word.append(i)
            wordw += widths[i]
            if c.endswith(' ') or i == len(chunks) - 1:
                if cur and curw + wordw > 940:
                    rows.append(cur)
                    cur, curw = [], 0.0
                cur += word
                curw += wordw
                word, wordw = [], 0.0
        rows.append(cur)
        lh = 76
        padx, pady = 34, 22
        maxw = max(sum(widths[i] for i in r) for r in rows)
        bw, bh = int(maxw + 2 * padx + 40), int(len(rows) * lh + 2 * pady + 40)
        base = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        ImageDraw.Draw(base).rounded_rectangle((20, 20, bw - 20, bh - 20), radius=38, fill=(20, 8, 4, 130))
        hi = base.copy()
        boxes = {}
        for ri, r in enumerate(rows):
            rw = sum(widths[i] for i in r)
            x = (bw - rw) / 2
            y = 20 + pady + ri * lh
            for i in r:
                for img, col in ((base, WHITE), (hi, GOLD)):
                    sh = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
                    ImageDraw.Draw(sh).text((x + 2, y + 4), chunks[i], font=fnt, fill=(0, 0, 0, 160))
                    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(4)))
                    ImageDraw.Draw(img).text((x, y), chunks[i], font=fnt, fill=col + (255,),
                                             stroke_width=4, stroke_fill=(40, 14, 0, 255))
                boxes[i] = (int(x - 4), int(y - 8), int(x + widths[i] + 4), int(y + lh))
                x += widths[i]
        LYR.append(dict(base=np.array(base), hi=np.array(hi), boxes=boxes, notes=ns,
                        t0=spans[li][0], t1=spans[li][1]))
    for L in LYR:
        L['vis0'] = L['t0'] - 0.6
        L['vis1'] = L['t1'] + 0.55
    for i in range(1, len(LYR)):
        if LYR[i]['vis0'] < LYR[i - 1]['vis1']:
            mid = max(LYR[i - 1]['t1'] + 0.1, LYR[i]['t0'] - 0.45)
            LYR[i - 1]['vis1'] = mid
            LYR[i]['vis0'] = mid


CAPTIONS = [  # zusätzliche Untertitel ohne Gesang (Start, Ende, Text)
    (6.6, 9.8, '(Billy wärmt die Stimme auf …)'),
    (36.4, 38.3, '(Gitarrensolo!)'),
    (40.6, 41.35, '(Gitarrensolo!)'),
]

# Große Wörter im Refrain (Zeit, Wort, Farbe)
BIGWORDS = [(28.0, 'OYE!', GOLD), (28.5, 'COMO', CYAN), (29.0, 'WAU!', PINK),
            (30.0, 'WAU!', GOLD), (30.25, 'WAU!', PINK),
            (32.0, 'LORD', GOLD), (32.5, 'BILLY', PINK), (33.0, 'SINGT', CYAN),
            (33.5, 'FÜR DICH', GOLD), (34.5, 'AUUU!', PINK)]


def credits_sprite():
    lines = [('Lord Billy', font(F_XBOLD, 46)), ('»Oye Como Wau«', font(F_SEMI, 40)),
             ('Löwenmähne', font(F_SEMI, 34)), ('Chow Chow Records', font(F_SEMI, 34))]
    w = int(max(f.getlength(s) for s, f in lines)) + 60
    hgt = 52 * len(lines) + 50
    img = Image.new('RGBA', (w, hgt), (0, 0, 0, 0))
    sh = Image.new('RGBA', (w, hgt), (0, 0, 0, 0))
    y = 20
    for s, f in lines:
        ImageDraw.Draw(sh).text((22, y + 3), s, font=f, fill=(0, 0, 0, 200))
        y += 52
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(5)))
    y = 20
    for s, f in lines:
        ImageDraw.Draw(img).text((20, y), s, font=f, fill=(255, 255, 255, 255))
        y += 52
    return np.array(img)


def bug_sprite():
    """Sender-Logo "WAU TV" (eigene Gestaltung) für die obere rechte Ecke."""
    img = Image.new('RGBA', (300, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((18, 14), 'WAU', font=font(F_BLACK, 92), fill=(255, 255, 255, 235),
           stroke_width=3, stroke_fill=(0, 0, 0, 120))
    d.rounded_rectangle((176, 104, 262, 150), radius=10, fill=(255, 60, 150, 240))
    d.text((186, 104), 'TV', font=font(F_BLACK, 40), fill=(255, 255, 255, 255))
    img = img.rotate(8, resample=Image.BICUBIC, expand=True)
    return np.array(img)


def premiere_sprite():
    img = Image.new('RGBA', (420, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, 416, 86), radius=14, fill=(255, 60, 150, 235))
    d.text((24, 14), 'WELTPREMIERE', font=font(F_BLACK, 50), fill=(255, 255, 255, 255))
    return np.array(img)


def build_sprites():
    SPR['mic'] = build_mic()
    SPR['title'] = text_sprite('LORD BILLY', font(F_BLACK, 150), GOLD, stroke=8, shadow=14)
    SPR['sub'] = text_sprite('Chow Chow · 6 Jahre', font(F_BOLD, 56), WHITE, stroke=4)
    SPR['sings'] = text_sprite('singt', font(F_ITAL, 52), WHITE, stroke=4)
    SPR['song'] = text_sprite('»Oye Como Wau«', font(F_BLACK, 92), ORANGE, stroke=6, stroke_fill=(50, 10, 0))
    SPR['style'] = text_sprite('Latin-Rock à la Santana', font(F_BOLD, 46), WHITE, stroke=4)
    SPR['solo'] = text_sprite('GITARRENSOLO!', font(F_BLACK, 112), PINK, stroke=7, stroke_fill=(40, 0, 20))
    SPR['end1'] = text_sprite('LORD BILLY', font(F_BLACK, 140), GOLD, stroke=8, shadow=14)
    SPR['end2'] = text_sprite('Danke fürs Zuhören – Wau!', font(F_ITAL, 50), WHITE, stroke=4)
    SPR['credits'] = credits_sprite()
    SPR['bug'] = bug_sprite()
    SPR['premiere'] = premiere_sprite()
    for i, (_, _, txt) in enumerate(CAPTIONS):
        SPR[f'cap{i}'] = text_sprite(txt, font(F_ITAL, 50), WHITE, stroke=4, shadow=8)
    SPR['big'] = [text_sprite(w, font(F_BLACK, 170), c, stroke=9, stroke_fill=(20, 0, 30), shadow=12)
                  for _, w, c in BIGWORDS]
    sym = font(F_SYM, 96)
    SPR['notes'] = []
    for g in '♪♫♩♬':
        for col in (GOLD, ORANGE, PINK, WHITE):
            SPR['notes'].append(text_sprite(g, sym, col, stroke=4, stroke_fill=(40, 10, 0), shadow=6))


def window(t, a, b, fade=0.25):
    return float(smooth((t - a) / fade) * smooth((b - t) / fade))


def draw_lyrics(F, t):
    for L in LYR:
        if not (L['vis0'] <= t < L['vis1']):
            continue
        al = window(t, L['vis0'], L['vis1'], 0.18)
        spr = L['base'].copy()
        for i, n in enumerate(L['notes']):
            p = np.clip((t - n['t0']) / max(n['t1'] - n['t0'], 0.12), 0, 1)
            if p <= 0:
                continue
            x0, y0, x1, y1 = L['boxes'][i]
            xe = int(x0 + p * (x1 - x0))
            spr[y0:y1, x0:xe] = L['hi'][y0:y1, x0:xe]
        rise = 18 * (1 - smooth((t - L['vis0']) / 0.25))
        blend(F, spr, W / 2, 1650 + rise, al)
    for i, (a, b, _) in enumerate(CAPTIONS):
        if a <= t < b:
            blend(F, SPR[f'cap{i}'], W / 2, 1660, window(t, a, b, 0.3))


def draw_bigwords(F, t):
    for i, (t0, _, _) in enumerate(BIGWORDS):
        t1 = BIGWORDS[i + 1][0] if i + 1 < len(BIGWORDS) and BIGWORDS[i + 1][0] - t0 < 1.0 else t0 + 0.9
        if not (t0 <= t < t1):
            continue
        a = t - t0
        sc = 1.0 + 0.55 * np.exp(-a / 0.06)
        ang = 6 if i % 2 == 0 else -6
        blend(F, xform(SPR['big'][i], sc, ang), W / 2, 330, smooth(a / 0.04) * smooth((t1 - t) / 0.1))


def draw_titles(F, t):
    # Intro
    if t < 4.2:
        a = window(t, 0.25, 3.85, 0.35)
        sc = 0.85 + 0.15 * smooth((t - 0.25) / 0.5) + 0.04 * kick_env(t)
        blend(F, xform(SPR['title'], sc), W / 2, 270, a)
        blend(F, SPR['sub'], W / 2, 400, window(t, 0.7, 3.85, 0.35))
        blend(F, SPR['sings'], W / 2, 1450, window(t, 1.5, 3.85, 0.3))
        blend(F, xform(SPR['song'], 0.9 + 0.1 * smooth((t - 1.8) / 0.4)), W / 2, 1545, window(t, 1.8, 3.85, 0.3))
        blend(F, SPR['style'], W / 2, 1650, window(t, 2.3, 3.85, 0.3))
        blink = 1.0 if int(t * 2.5) % 2 == 0 else 0.55
        blend(F, SPR['premiere'], 60 + SPR['premiere'].shape[1] / 2, 110, window(t, 0.3, 3.9, 0.2) * blink)
    # Sender-Logo
    if 0.6 <= t < 44.6:
        b = SPR['bug']
        blend(F, b, W - 40 - b.shape[1] / 2, 50 + b.shape[0] / 2, 0.8 * window(t, 0.6, 44.6, 0.4))
    # Credits unten links (wie im Musikfernsehen: am Anfang und am Ende)
    c = SPR['credits']
    for a, bnd in ((4.4, 9.6), (42.4, 44.8)):
        if a <= t < bnd:
            blend(F, c, 40 + c.shape[1] / 2, 1330 + c.shape[0] / 2, window(t, a, bnd, 0.35))
    # Solo
    for a, b in ((36.0, 39.0), (40.0, 42.0)):
        if a <= t < b:
            sc = 1.0 + 0.1 * kick_env(t)
            blend(F, xform(SPR['solo'], sc, 4 * np.sin(t * 6)), W / 2, 300, window(t, a, b, 0.15))
    # Abspann
    if t >= 42.5:
        blend(F, SPR['end1'], W / 2, 320, window(t, 42.5, 50, 0.5))
        blend(F, SPR['end2'], W / 2, 450, window(t, 43.0, 50, 0.5))


# ----------------------------------------------------------------- Noten-Partikel
PARTICLES = []


def build_particles():
    rng = np.random.default_rng(6)
    for n in TL.vocal_notes():
        cnt = 1 + int(n['t1'] - n['t0'] > 0.6) + int(n['t1'] - n['t0'] > 1.2)
        for j in range(cnt):
            ts = n['t0'] + j * 0.35
            sh = shot_at(ts)
            if 'split' in sh:
                xc, y, half = W / 2, H / 2, 150
            else:
                g = mouth_geom(sh['img'], camera(sh, ts))
                xc, y, half = g['xc'], g['lip'] + g['J'] * 0.2, g['half']
            side = rng.choice([-1, 1])
            PARTICLES.append(dict(
                t0=ts, life=rng.uniform(1.4, 2.0), spr=int(rng.integers(len(SPR['notes']))),
                x=xc + side * half * rng.uniform(0.7, 1.1), y=y,
                vx=side * rng.uniform(90, 190), vy=-rng.uniform(260, 380),
                sc=rng.uniform(0.6, 1.0), ph=rng.uniform(0, 6.28)))


def draw_particles(F, t):
    for p in PARTICLES:
        a = t - p['t0']
        if a < 0 or a > p['life']:
            continue
        x = p['x'] + p['vx'] * a + 22 * np.sin(4 * a + p['ph'])
        y = p['y'] + p['vy'] * a + 40 * a * a
        al = smooth(a / 0.12) * smooth((p['life'] - a) / 0.5)
        sc = p['sc'] * (0.7 + 0.3 * smooth(a / 0.3))
        blend(F, xform(SPR['notes'][p['spr']], sc), x, y, al)


# ----------------------------------------------------------------- Bild komponieren
MOUTH = None


def init_worker():
    global MOUTH
    load_images()
    build_sprites()
    build_lyrics()
    build_particles()
    build_grain()
    MOUTH = TL.mouth_open(np.arange(NF) / FPS)


def render(fi):
    t = fi / FPS
    sh = shot_at(t)
    sec = TL.section_at(t)
    photo = photo_frame(t, float(MOUTH[fi]))
    F = photo.astype(np.float32) / 255.0
    # Farbstimmung je Abschnitt
    if 'tint' in sh:
        hi, lo = TINTS[sh['tint']]
        F = tone(F, hi, lo, 0.24)
    elif sec == 'chorus':
        F = tone(F, *CHORUS_TONE, 0.24)
    mult, add = light_maps(t)
    F *= mult
    F += add * (1 - F)
    # Stroboskop / Blitze
    if sec == 'chorus':
        snares = np.array([TL.bt(b, k) for b in range(14, 18) for k in (1, 3)])
        F += (1 - F) * 0.22 * env_after(snares, t, 0.05)
    if sec == 'solo':
        col = np.array(TINTS[int(t) % 3][0], np.float32)
        F += (col - F) * 0.16 * env_after(BEATS, t, 0.06)
    for s in SHOTS:
        if s['flash'] and 0 <= t - s['t0'] < 0.25:
            F += (1 - F) * 0.55 * np.exp(-(t - s['t0']) / 0.07)
    # RGB-Versatz auf Kicks und Schnitten
    cut = max([np.exp(-(t - s['t0']) / 0.1) for s in SHOTS if 0 <= t - s['t0'] < 0.4] + [0])
    d = int(round((14 if sec in ('chorus', 'solo') else 6) * kick_env(t) + 10 * cut))
    if d > 0:
        F[:, d:, 0] = F[:, :-d, 0].copy()
        F[:, :-d, 2] = F[:, d:, 2].copy()
    # Filmkorn
    g = cv2.resize(GRAIN[(fi // 3) % len(GRAIN)], (W, H), interpolation=cv2.INTER_LINEAR)
    F += g[..., None]
    np.clip(F, 0, 1, out=F)
    draw_particles(F, t)
    draw_bigwords(F, t)
    draw_titles(F, t)
    draw_lyrics(F, t)
    F *= smooth(t / 0.5) * smooth((TL.DURATION - t) / 0.8)
    return (np.clip(F, 0, 1) * 255 + 0.5).astype(np.uint8)


def write_srt(path):
    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    items = [(L['vis0'], L['vis1'], ''.join(n['text'] for n in L['notes']).strip()) for L in LYR]
    items += [(a, b, txt) for a, b, txt in CAPTIONS]
    items.sort()
    with open(path, 'w', encoding='utf-8') as f:
        for i, (a, b, txt) in enumerate(items, 1):
            f.write(f"{i}\n{ts(max(a, 0))} --> {ts(b)}\n{txt}\n\n")


def main():
    if sys.argv[1] == '--preview':
        times = [float(x) for x in sys.argv[2].split(',')]
        out = sys.argv[3]
        os.makedirs(out, exist_ok=True)
        init_worker()
        for t in times:
            fr = render(int(round(t * FPS)))
            Image.fromarray(fr).save(os.path.join(out, f'frame_{t:05.2f}.jpg'), quality=90)
        return
    wav, mp4 = sys.argv[1], sys.argv[2]
    init_worker()
    write_srt(os.path.splitext(mp4)[0] + '.srt')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-i', wav,
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-maxrate', '9M', '-bufsize', '18M', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker) as pool:
        for i, fr in enumerate(pool.imap(render, range(NF), chunksize=4)):
            proc.stdin.write(fr.tobytes())
            if i % 150 == 0:
                print(f'  Bild {i}/{NF}', flush=True)
    proc.stdin.close()
    proc.wait()
    print('fertig:', mp4)


if __name__ == '__main__':
    main()
