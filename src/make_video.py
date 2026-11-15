"""Rendert das Musikvideo "Lord Billy – Oye Como Wau" im Stil des 80er/90er-Musikfernsehens.

Billy ist aus den drei Fotos freigestellt (siehe make_cutouts.py) und steht
vor grafischen Neon-Hintergründen: Sonnenstrahlen, Memphis-Formen, Bühne
mit Scheinwerfern, Hypno-Tunnel, Laufschrift und ein Pop-Art-Raster mit
sechs Billys. Sein Unterkiefer bewegt sich synchron zu den gesungenen
Phonemen, vor dem Maul steht ein Mikrofon. Dazu Sender-Logo, Credits unten
links, VHS-Rauschen bei Schnitten, RGB-Versatz, große Wörter im Refrain und
Karaoke-Untertitel der gerade gesungenen Zeile.

Aufruf:
  python3 src/make_video.py build/song.wav video/LordBilly_OyeComoWau.mp4
  python3 src/make_video.py --preview 13.2,22.0 out_dir     # Einzelbilder
  python3 src/make_video.py --ohne-ton video/LordBilly_ohneTon.mp4

Die Fassung --ohne-ton hat keine Tonspur, keine Liedtext-Untertitel und keine
taktgenauen Effekte. Billy singt darin frei (Maul, Kopf, Augen), damit man in
Instagram oder TikTok einen Song aus deren lizenzierter Musikbibliothek
darunterlegen kann.
"""
import json
import os
import sys
import subprocess
from multiprocessing import Pool

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import timeline as TL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREE = os.environ.get('LB_OHNE_TON') == '1'   # Fassung ohne Ton (siehe oben)
W, H = 1080, 1920          # Rendergröße
OUT_W, OUT_H = 720, 1280   # Größe des fertigen Videos
TARGET_MB = 18.5           # Zielgröße der MP4-Datei
FPS = TL.FPS
NF = int(round(TL.DURATION * FPS))

FONT_DIR = '/usr/share/fonts/opentype/inter/'
F_BLACK = FONT_DIR + 'InterDisplay-Black.otf'
F_XBOLD = FONT_DIR + 'InterDisplay-ExtraBold.otf'
F_BOLD = FONT_DIR + 'InterDisplay-Bold.otf'
F_SEMI = FONT_DIR + 'InterDisplay-SemiBold.otf'
F_ITAL = FONT_DIR + 'InterDisplay-SemiBoldItalic.otf'
F_SYM = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# Neon-Palette
PINK = (255, 46, 136)
YELLOW = (255, 214, 0)
CYAN = (0, 200, 255)
PURPLE = (120, 60, 255)
GREEN = (0, 230, 118)
ORANGE = (255, 120, 0)
BLACK = (18, 14, 28)
WHITE = (255, 255, 255)
GOLD = (255, 200, 61)

# Maul-Geometrie und Mikrofon je Foto (Pixel im Originalfoto)
GEOM = {
    'A': dict(mouth=dict(x=1470, lip=1345, gap=70, jaw=1700, decay=420, half=330, slope=0.0, amax=115),
              mic=dict(x=1745, y=1745, r=150, tilt=30)),
    'B': dict(mouth=dict(x=965, lip=842, gap=42, jaw=975, decay=230, half=180, slope=0.06, amax=58),
              mic=dict(x=1160, y=1035, r=80, tilt=32)),
    'C': dict(mouth=dict(x=521, lip=627, gap=9, jaw=683, decay=95, half=52, slope=0.0, amax=19),
              mic=dict(x=592, y=716, r=27, tilt=30)),
}
# Augen (Mitte x, y, halbe Breite, halbe Höhe), Kopf (Mitte, Radius, Drehpunkt am Hals), Pfoten
EYES = {
    'A': [(1127, 790, 80, 43), (1677, 757, 77, 43)],
    'B': [(877, 528, 52, 30), (1100, 522, 22, 18)],
    'C': [(499, 561, 10, 6), (562, 567, 10, 6)],
}
HEAD = {'A': dict(c=(1460, 1080), R=820, pivot=(1460, 2050)),
        'B': dict(c=(880, 640), R=470, pivot=(830, 1250)),
        'C': dict(c=(552, 615), R=125, pivot=(552, 790))}
FEET = {'A': (1410, 3950), 'B': (816, 2000), 'C': (550, 1030)}
# Bildausschnitte: (Quellpunkt x, y, Quellpixel je Ausgabepixel, Ankerpunkt im Bild x, y)
FRAMES = {
    ('A', 'full'): (1410, 2040, 2.75, 540, 1010),
    ('A', 'close'): (1460, 1150, 1.55, 540, 840),
    ('B', 'full'): (816, 1056, 1.35, 540, 1000),
    ('B', 'close'): (860, 700, 0.95, 540, 820),
    ('C', 'full'): (550, 760, 0.45, 540, 1010),
}
TILE_FRAMES = {'A': (1460, 1150, 3.0, 270, 330), 'B': (860, 700, 1.85, 270, 320)}

# Einstellungen
SHOTS = [
    dict(t0=0.0, t1=4.0, bg='sunburst', pal=(YELLOW, ORANGE), img='C', fr='full', z=(0.85, 1.0), flash=False),
    dict(t0=4.0, t1=6.0, bg='memphis', pal=(PINK, YELLOW, CYAN), img='B', fr='full', z=(1.0, 1.06)),
    dict(t0=6.0, t1=8.0, bg='marquee', pal=(CYAN, PINK), img='A', fr='full', z=(1.0, 1.08)),
    dict(t0=8.0, t1=12.0, bg='stage', img='B', fr='close', z=(0.78, 1.0)),
    dict(t0=12.0, t1=16.0, bg='memphis', pal=(YELLOW, PINK, CYAN), img='A', fr='close', z=(0.95, 1.05), vhs=True),
    dict(t0=16.0, t1=20.0, bg='sunburst', pal=(PINK, PURPLE), img='B', fr='close', z=(0.95, 1.04)),
    dict(t0=20.0, t1=24.0, bg='tunnel', pal=(CYAN, PURPLE, PINK), img='A', fr='close', z=(0.95, 1.05), rot=(-4, 3)),
    dict(t0=24.0, t1=28.0, bg='stage', img='C', fr='full', z=(1.0, 1.12), vhs=True),
    dict(t0=28.0, t1=30.0, bg='sunburst', pal=(YELLOW, PINK), img='A', fr='close', z=(1.0, 1.06)),
    dict(t0=30.0, t1=32.0, bg='grid', imgs='ABBAAB'),
    dict(t0=32.0, t1=34.0, bg='marquee', pal=(PINK, YELLOW), img='B', fr='close', z=(1.0, 1.06)),
    dict(t0=34.0, t1=36.0, bg='tunnel', pal=(YELLOW, ORANGE, PINK), img='A', fr='close', z=(1.0, 1.1), rot=(0, -3)),
    dict(t0=36.0, t1=37.0, bg='stage', img='B', fr='full', z=(1.0, 1.05), vhs=True),
    dict(t0=37.0, t1=38.0, bg='memphis', pal=(CYAN, YELLOW, PINK), img='C', fr='full', z=(1.0, 1.06)),
    dict(t0=38.0, t1=39.0, bg='grid', imgs='BAABBA'),
    dict(t0=39.0, t1=40.0, bg='sunburst', pal=(PINK, YELLOW), img='A', fr='close', z=(1.0, 1.08)),
    dict(t0=40.0, t1=41.0, bg='tunnel', pal=(GREEN, PURPLE, YELLOW), img='B', fr='close', z=(1.0, 1.06)),
    dict(t0=41.0, t1=42.0, bg='marquee', pal=(YELLOW, PURPLE), img='C', fr='full', z=(1.0, 1.06)),
    dict(t0=42.0, t1=45.0, bg='stage', img='A', fr='full', z=(1.2, 0.95), vhs=True),
]
for _s in SHOTS:
    _s.setdefault('flash', True)
GRID_COLORS = [(PINK, YELLOW), (CYAN, PINK), (YELLOW, PURPLE), (GREEN, PINK), (PURPLE, YELLOW), (ORANGE, CYAN)]


# ----------------------------------------------------------------- Hilfen
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(x):
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def col(c):
    return np.array(c, np.float32) / 255


KICKS = TL.kick_times()
BEATS = np.array([TL.bt(b, k) for b in range(2, 21) for k in range(4)] + [TL.bt(21)])


def env_after(times, t, tau):
    past = times[times <= t]
    if len(past) == 0:
        return 0.0
    return float(np.exp(-(t - past[-1]) / tau))


def kick_env(t):
    if FREE:
        return 0.0
    return env_after(KICKS, t, 0.13)


def beat_index(t):
    return int(np.sum(BEATS <= t))


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


# ----------------------------------------------------------------- Billy (freigestellt)
_CUT = {}


def load_cutouts():
    meta = json.load(open(os.path.join(ROOT, 'bilder', 'freigestellt', 'cutouts.json')))
    for k, m in meta.items():
        a = np.asarray(Image.open(os.path.join(ROOT, m['file']))).astype(np.float32)
        rgb, al = a[..., :3], a[..., 3:4] / 255
        # knackiger Musikvideo-Look
        rgb = 255 * (0.5 + 1.1 * (rgb / 255 - 0.5))
        rgb[..., 0] *= 1.03
        rgb = np.clip(rgb, 0, 255)
        pre = np.dstack([rgb * al, al[..., 0] * 255]).astype(np.uint8)   # vormultipliziert
        _CUT[k] = dict(img=pre, f=m['scale'])


def make_cam(k, fr, t0, t1, t, z=(1.0, 1.0), rot=(0.0, 0.0), scale=1.0, anchor=None):
    """Kamera als affine Abbildung Foto -> Bild (2x3-Matrix M), inkl. Stauchen auf dem Beat."""
    cx, cy, s, ax, ay = TILE_FRAMES[k] if fr == 'tile' else FRAMES[(k, fr)]
    if anchor is not None:
        ax, ay = anchor
    u = ease((t - t0) / (t1 - t0))
    zz = z[0] + (z[1] - z[0]) * u
    r = rot[0] + (rot[1] - rot[0]) * u
    sec = TL.section_at(t)
    hot = sec in ('chorus', 'solo')
    ke = kick_env(t)
    zz *= 1 + (0.045 if hot else 0.02) * ke
    r += (3.0 if hot else 1.6) * np.sin(2 * np.pi * t / (2 * TL.BEAT * 2))   # Wippen im Takt
    ay -= (22 if hot else 10) * ke * (1 if fr != 'tile' else 0.4)              # kleiner Hüpfer auf dem Kick
    s_eff = s / zz * scale
    th = np.radians(r)
    c, sn = np.cos(th), np.sin(th)
    L = np.array([[c, sn], [-sn, c]]) / s_eff
    tr = np.array([ax, ay]) - L @ np.array([cx, cy])
    # Squash & Stretch um die Pfoten: auf dem Kick kurz gestaucht, danach gestreckt
    sq = (0.05 if hot else 0.03) * ke - (0 if FREE else 0.012 * np.exp(-((t % TL.BEAT) - 0.22) ** 2 / 0.006))
    S = np.diag([1 + 0.5 * sq, 1 - sq])
    fo = L @ np.array(FEET[k]) + tr
    L2 = S @ L
    t2 = fo + S @ (tr - fo)
    return dict(M=np.hstack([L2, t2[:, None]]), s=s_eff, rot=r, sx=S[0, 0], sy=S[1, 1])


def to_out(cam, q):
    q = np.asarray(q, dtype=float)
    return cam['M'][:, :2] @ q + cam['M'][:, 2]


def render_cutout(k, cam, size=(W, H)):
    d = _CUT[k]
    m = cam['M'].copy()
    m[:, :2] /= d['f']
    return cv2.warpAffine(d['img'], m, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def mouth_geom(k, cam):
    m = GEOM[k]['mouth']
    sv = cam['s'] / cam['sy']
    sh = cam['s'] / cam['sx']
    xc, lip = to_out(cam, (m['x'], m['lip']))
    slope = np.tan(np.arctan(m['slope']) - np.radians(cam['rot']))
    return dict(xc=xc, lip=lip, gap=m['gap'] / sv, J=(m['jaw'] - m['lip']) / sv, D=m['decay'] / sv,
                half=m['half'] / sh, slope=slope, amax=1.25 * m['amax'] / sv)


# ----------------------------------------------------------------- Bewegung: Kopf und Augen
POSE = {}
FREE_OPEN = {'a': 1.0, 'o': 0.8, 'e': 0.65, 'u': 0.5, 'au': (1.0, 0.5), 'ai': (1.0, 0.45)}


def free_notes():
    """Frei gesungene Silben (ohne festen Song): Phrasen mit Pausen und langen Schlusstönen."""
    rng = np.random.default_rng(21)
    notes, t = [], 4.6
    vowels = ['a', 'a', 'o', 'o', 'e', 'u', 'au', 'ai', 'a', 'o']
    while t < 43.0:
        end = min(t + rng.uniform(2.4, 4.2), 43.6)
        while t < end - 0.15:
            d = float(rng.choice([0.2, 0.25, 0.3, 0.35, 0.45, 0.6]))
            if end - t < 1.9 and rng.random() < 0.6:
                d = end - t
            notes.append(dict(t0=t, t1=t + d, midi=int(rng.integers(64, 80)),
                              vowel=vowels[int(rng.integers(len(vowels)))], text=''))
            t += d + rng.uniform(0.02, 0.08)
        t = end + rng.uniform(0.5, 1.2)
    return notes


def notes_list():
    return free_notes() if FREE else TL.vocal_notes()


def free_mouth(t):
    from scipy import signal
    grid = np.arange(0, TL.DURATION + 0.5, 0.001)
    val = np.zeros_like(grid)
    for n in free_notes():
        i0, i1 = int(n['t0'] * 1000), int(n['t1'] * 1000)
        m = FREE_OPEN[n['vowel']]
        x = np.linspace(0, 1, i1 - i0)
        v = m[0] + (m[1] - m[0]) * np.clip((x - 0.35) / 0.5, 0, 1) if isinstance(m, tuple) else np.full(i1 - i0, m)
        v[:40] = 0.15                      # kurzer Konsonant am Silbenanfang
        val[i0:i1] = v
    val = signal.sosfiltfilt(signal.butter(2, 9, fs=1000, output='sos'), val)
    return np.clip(np.interp(t, grid, val), 0, 1.05)


def compute_pose():
    """Kopfbewegung und Augen je Videobild aus Beat und Gesang."""
    from scipy import signal
    t = np.arange(NF) / FPS
    theta, lift, scl, vib = (np.zeros(NF) for _ in range(4))
    if FREE:                             # ohne festen Takt: ruhiges Wiegen
        theta += 2.2 * np.sin(2 * np.pi * 0.45 * t)
        lift += 0.012 * np.sin(2 * np.pi * 0.9 * t)
    else:
        for i, b in enumerate(BEATS):    # Nicken im Takt, abwechselnd links/rechts
            e = np.where(t >= b, np.exp(-(t - b) / 0.18), 0)
            theta += 2.4 * (1 if i % 2 == 0 else -1) * e
            lift += 0.018 * e
    eyes = np.zeros(NF)
    for j, n in enumerate(notes_list()):
        d = n['t1'] - n['t0']
        a = t - n['t0']
        on = (a >= 0) & (t < n['t1'] + 0.15)
        side = 1 if j % 2 else -1
        lift += np.where(on, -0.03 * np.exp(-a / 0.12), 0)                 # Ansatz jeder Silbe
        theta += np.where(on, 1.8 * side * np.exp(-a / 0.2), 0)
        lift += np.where(on, -0.035 * (n['midi'] - 72) / 12 * smooth(a / 0.15), 0)   # hohe Töne: Kopf hoch
        if d > 0.7:                                                          # lange Töne: Kopf in den Nacken
            p = smooth((a - 0.15) / 0.5) * smooth((n['t1'] + 0.1 - t) / 0.25)
            theta += np.where(on, side * 6 * p, 0)
            lift += np.where(on, -0.06 * p, 0)
            scl += np.where(on, 0.03 * p, 0)
            vib = np.maximum(vib, np.where(on, p, 0))
        if d > 0.9:                                                          # Augen zu vor Hingabe
            p = smooth((a - 0.3) / 0.3) * smooth((n['t1'] - 0.05 - t) / 0.2)
            eyes = np.maximum(eyes, np.where(on, 0.85 * p, 0))
    sos = signal.butter(2, 4, fs=FPS, output='sos')
    theta, lift, scl = (signal.sosfiltfilt(sos, x) for x in (theta, lift, scl))
    theta += 1.3 * np.sin(2 * np.pi * 5.3 * t) * np.clip(vib, 0, 1)        # Vibrato-Wackeln
    rng = np.random.default_rng(11)
    tb = 1.2
    while tb < TL.DURATION:                                                  # natürliches Blinzeln
        eyes = np.maximum(eyes, np.clip(1 - np.abs(t - tb) / 0.09, 0, 1))
        tb += rng.uniform(2.2, 4.0)
    POSE.update(theta=theta, lift=lift, scl=scl, eyes=eyes)


def eye_close(layer, cam, k, c):
    """Augenlider schließen: Fell über dem Auge wird nach unten gezogen."""
    if c < 0.03:
        return layer
    fh, fw = layer.shape[:2]
    for ex, ey, ew, eh in EYES[k]:
        e = to_out(cam, (ex, ey))
        ew_o, eh_o = ew / cam['s'], eh / cam['s']
        if ew_o < 3:
            continue
        band = eh_o * 1.6
        x0, x1 = int(max(e[0] - ew_o, 0)), int(min(e[0] + ew_o, fw - 1))
        y0, y1 = int(max(e[1] - eh_o - band - 2, 0)), int(min(e[1] + eh_o + 2, fh - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        Y = np.arange(y0, y1 + 1, dtype=np.float32)[:, None]
        prof = np.sqrt(np.clip(1 - ((xs - e[0]) / ew_o) ** 2, 0, 1))[None]
        top = e[1] - eh_o * prof
        lid = top + c * 2 * eh_o * prof * 0.95
        cover = (Y < lid) & (Y > top - band)
        ysrc = (top - band) + (Y - (top - band)) * band / (band + (lid - top))
        my = np.where(cover, ysrc, Y).astype(np.float32)
        mx = np.broadcast_to(xs[None], my.shape).copy()
        patch = cv2.remap(layer, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101).astype(np.float32)
        lash = np.exp(-((Y - lid) / 1.6) ** 2) * (prof > 0.15) * min(1, c * 1.5)
        patch[..., :3] *= (1 - 0.5 * lash[..., None])
        layer[y0:y1 + 1, x0:x1 + 1] = patch.astype(np.uint8)
    return layer


def head_warp(layer, cam, k, fi):
    """Dreht, hebt und skaliert den Kopf um den Hals (weich ausgeblendet zum Körper)."""
    hd = HEAD[k]
    th = np.radians(POSE['theta'][fi])
    sc = 1 + POSE['scl'][fi]
    R = hd['R'] / cam['s']
    dy = POSE['lift'][fi] * R
    if abs(th) < 1e-3 and abs(dy) < 0.3 and abs(sc - 1) < 1e-3:
        return layer
    fh, fw = layer.shape[:2]
    hc, pv = to_out(cam, hd['c']), to_out(cam, hd['pivot'])
    rr = 1.75 * R
    x0, x1 = int(max(hc[0] - rr, 0)), int(min(hc[0] + rr, fw - 1))
    y0, y1 = int(max(hc[1] - rr, 0)), int(min(hc[1] + rr, fh - 1))
    if x1 <= x0 or y1 <= y0:
        return layer
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    d = np.sqrt((xx - hc[0]) ** 2 + (yy - hc[1]) ** 2)
    w = 1 - smooth((d - R) / (0.75 * R))
    ang = th * w
    c, s = np.cos(ang), np.sin(ang)
    scale = 1 + (sc - 1) * w
    X = xx - pv[0]
    Y = yy - pv[1] - dy * w
    mx = (pv[0] + (c * X + s * Y) / scale).astype(np.float32)
    my = (pv[1] + (-s * X + c * Y) / scale).astype(np.float32)
    patch = cv2.remap(layer, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out = layer.copy()
    out[y0:y1 + 1, x0:x1 + 1] = patch
    return out


def animate(k, cam, size, fi):
    """Billy rendern: Augen, Maul, Kopfbewegung."""
    lay = render_cutout(k, cam, size)
    lay = eye_close(lay, cam, k, float(POSE['eyes'][fi]))
    lay = open_mouth(lay, mouth_geom(k, cam), float(MOUTH[fi]))
    return head_warp(lay, cam, k, fi)


def open_mouth(frame, g, amount):
    """Öffnet den Unterkiefer: Bereich unter der Oberlippe wird nach unten gedehnt."""
    A = amount * g['amax']
    if A < 0.4:
        return frame
    fh, fw = frame.shape[:2]
    hw = 1.3 * g['half']
    x0, x1 = int(max(g['xc'] - hw, 0)), int(min(g['xc'] + hw, fw - 1))
    span = g['J'] + g['D'] + 4
    if x1 <= x0:
        return frame
    offs = g['slope'] * (np.arange(x0, x1 + 1) - g['xc'])
    y0 = int(max(g['lip'] + offs.min() - 6, 0))
    y1 = int(min(g['lip'] + offs.max() + span + 6, fh - 1))
    if y1 <= y0:
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


def composite_billy(F, layer, outline=10, shadow=(18, 22), shadow_col=BLACK, posterize=None):
    """Billy-Ebene (vormultipliziert, RGBA uint8) mit Sticker-Kontur und Pop-Schatten auf F legen."""
    a = layer[..., 3].astype(np.float32) / 255
    rgb = layer[..., :3].astype(np.float32) / 255
    if posterize is not None:   # Pop-Art: Helligkeit -> drei Farbstufen
        dark, mid, light = posterize
        l = np.where(a > 0.01, rgb.mean(axis=2) / np.maximum(a, 0.01), 0)
        q = np.where(l[..., None] < 0.42, col(dark), np.where(l[..., None] < 0.72, col(mid), col(light)))
        rgb = (0.72 * q + 0.28 * rgb / np.maximum(a, 0.01)[..., None]) * a[..., None]
    if shadow:
        dx, dy = shadow
        sh = np.zeros_like(a)
        sh[dy:, dx:] = a[:-dy, :-dx]
        F *= (1 - 0.9 * sh[..., None])
        F += col(shadow_col) * 0.9 * sh[..., None]
    if outline:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline + 1, 2 * outline + 1))
        o = cv2.dilate((a > 0.35).astype(np.uint8) * 255, k)
        o = cv2.GaussianBlur(o, (0, 0), 1.2).astype(np.float32) / 255
        F *= (1 - o[..., None])
        F += o[..., None]
    F *= (1 - a[..., None])
    F += rgb


# ----------------------------------------------------------------- Mikrofon
MIC_R = 170


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
    c = np.where(is_ring[..., None], chrome, body)
    a_h = np.clip(hw - np.abs(xx - cx) + 0.5, 0, 1) * ((u >= 0) & (u <= 1))
    rgb = rgb * (1 - a_h[..., None]) + c * a_h[..., None]
    alpha = np.maximum(alpha, a_h)
    kx, ky, kr = cx, y1 + 0.18 * R, 0.24 * R
    d = np.sqrt((xx - kx) ** 2 + (yy - ky) ** 2)
    a_k = np.clip(kr - d + 0.5, 0, 1)
    nzk = np.sqrt(np.clip(1 - (d / kr) ** 2, 0, 1))
    dk = np.clip(((xx - kx) / kr) * L[0] + ((yy - ky) / kr) * L[1] + nzk * L[2], 0, 1)
    ck = np.array([0.1, 0.1, 0.11]) + dk[..., None] * 0.35
    rgb = rgb * (1 - a_k[..., None]) + ck * a_k[..., None]
    alpha = np.maximum(alpha, a_k)
    dx, dy = (xx - cx) / R, (yy - cy) / R
    r2 = dx ** 2 + dy ** 2
    inside = r2 <= 1.0
    nzs = np.sqrt(np.clip(1 - r2, 0, 1))
    dif = np.clip(dx * L[0] + dy * L[1] + nzs * L[2], 0, 1)
    spec = np.clip(2 * dif * nzs - L[2], 0, 1) ** 25
    lat = np.arcsin(np.clip(dy, -1, 1))
    lon = np.arctan2(dx, np.maximum(nzs, 1e-3))
    g1 = np.abs(np.sin(15 * (lat + lon)))
    g2 = np.abs(np.sin(15 * (lat - lon)))
    wire = smooth((0.35 - np.minimum(g1, g2)) / 0.2)
    hole = np.array([0.1, 0.1, 0.11]) + 0.12 * dif[..., None]
    metal = np.array([0.62, 0.63, 0.66]) * (0.35 + 0.75 * dif[..., None]) + spec[..., None]
    gcol = hole * (1 - wire[..., None]) + metal * wire[..., None] + spec[..., None] * 0.35
    gcol *= (0.55 + 0.45 * nzs[..., None] ** 0.5)
    band = (dy > 0.55) & (dy < 0.75)
    gcol = np.where(band[..., None], np.array([0.7, 0.71, 0.74]) * (0.4 + 0.7 * dif[..., None]) + spec[..., None], gcol)
    a_s = np.clip((1 - np.sqrt(r2)) * R + 0.5, 0, 1) * inside
    rgb = rgb * (1 - a_s[..., None]) + gcol * a_s[..., None]
    alpha = np.maximum(alpha, a_s)
    spr = np.dstack([np.clip(rgb, 0, 1) * 255, alpha * 255]).astype(np.uint8)
    return spr, (cx, cy), (kx, ky)


def draw_mic(F, k, cam, t):
    """Mikrofon samt Ständer passend zur Kamera vor das Maul setzen (F: float32)."""
    spec = GEOM[k]['mic']
    spr, (cx, cy), (kx, ky) = SPR['mic']
    fh, fw = F.shape[:2]
    gx, gy = to_out(cam, (spec['x'], spec['y']))
    r_out = spec['r'] / cam['s']
    sc = r_out / MIC_R
    ang = spec['tilt'] - cam['rot'] + 1.5 * np.sin(2 * np.pi * t / 3.1)
    m = cv2.getRotationMatrix2D((cx, cy), ang, sc)
    m[0, 2] += gx - cx
    m[1, 2] += gy - cy
    jx, jy = m @ np.array([kx, ky, 1.0])
    h, w = spr.shape[:2]
    pts = m @ np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    x0, x1 = int(max(pts[0].min() - 4, 0)), int(min(pts[0].max() + 4, fw))
    y0, y1 = int(max(min(pts[1].min(), jy) - 4, 0)), fh
    if x1 <= x0 or y1 <= y0:
        return
    m2 = m.copy()
    m2[0, 2] -= x0
    m2[1, 2] -= y0
    layer = cv2.warpAffine(spr, m2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderValue=(0, 0, 0, 0)).astype(np.float32) / 255
    pw = max(3, int(0.17 * r_out))
    px0 = int(jx - pw / 2) - x0
    if 0 <= px0 and px0 + pw <= layer.shape[1] and jy < fh:
        prof = np.linspace(-1, 1, pw)
        shade = 0.35 + 0.55 * np.clip(1 - (prof + 0.35) ** 2, 0, 1) + 0.4 * np.exp(-((prof + 0.3) / 0.15) ** 2)
        c = np.clip(np.outer(shade, [0.75, 0.76, 0.8]), 0, 1)
        ys = int(max(jy - y0, 0))
        pole = layer[ys:, px0:px0 + pw]
        under = pole[..., 3:4]
        pole[..., :3] = pole[..., :3] * under + c[None] * (1 - under)
        pole[..., 3] = 1
    a = layer[..., 3:4]
    reg = F[y0:y1, x0:x1]
    sh = cv2.GaussianBlur(a[..., 0], (0, 0), max(2, r_out * 0.15))
    off = max(1, int(r_out * 0.25))
    shm = np.zeros_like(sh)
    shm[off:, off:] = sh[:-off, :-off]
    reg *= (1 - 0.4 * shm[..., None])
    reg *= (1 - a)
    reg += layer[..., :3] * a


# ----------------------------------------------------------------- Hintergründe
_hy, _hx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32) * 2


def up(img):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def bg_sunburst(sh, t):
    c1, c2 = col(sh['pal'][0]), col(sh['pal'][1])
    ang = np.arctan2(_hy - 820, _hx - 540)
    r = np.sqrt((_hx - 540) ** 2 + (_hy - 820) ** 2)
    v = np.sin(16 * (ang + 0.9 * t))
    m = smooth((v + 0.08) / 0.16)[..., None]
    img = c1 * (1 - m) + c2 * m
    img *= (1.05 - 0.35 * smooth(r / 1300))[..., None]
    glow = np.exp(-(r / 380) ** 2)[..., None] * (0.35 + 0.25 * kick_env(t))
    img = img + (1 - img) * glow
    return up(img)


def bg_tunnel(sh, t):
    cs = [col(c) for c in sh['pal']]
    dx, dy = _hx - 540, _hy - 820
    r = np.maximum(np.abs(dx), np.abs(dy) * 0.8) + 1
    band = 5.0 * np.log(r) - 2.2 * t * (1 + 0.6 * kick_env(t))
    idx = np.floor(band).astype(int) % len(cs)
    frac = band - np.floor(band)
    edge = smooth(frac / 0.06)[..., None]   # Kantenglättung
    cur = np.stack(cs)[idx]
    prev = np.stack(cs)[(idx - 1) % len(cs)]
    img = prev * (1 - edge) + cur * edge
    img *= (0.7 + 0.3 * smooth(r / 900))[..., None]
    return up(img)


MEMPHIS = {}


def memphis_shapes(seed, pal):
    rng = np.random.default_rng(seed)
    kinds = ['circle', 'ring', 'triangle', 'zigzag', 'squiggle', 'dots', 'cross', 'ring', 'triangle', 'zigzag',
             'circle', 'squiggle', 'dots', 'cross']
    shapes = []
    for i, kd in enumerate(kinds):
        shapes.append(dict(kind=kd, x=rng.uniform(-50, W + 50), y=rng.uniform(-50, H + 50),
                           size=rng.uniform(70, 170), rot=rng.uniform(0, 360), vr=rng.uniform(-40, 40),
                           vx=rng.uniform(-30, 30), vy=rng.uniform(-40, 40),
                           col=pal[1 + i % (len(pal) - 1)] if i % 4 else (WHITE if i % 8 else BLACK)))
    return shapes


def bg_memphis(sh, t):
    key = id(sh)
    if key not in MEMPHIS:
        MEMPHIS[key] = memphis_shapes(int(sh['t0'] * 10), sh['pal'])
    # Grundfarbe wechselt auf jedem Takt zwischen den ersten beiden Palettenfarben
    base = sh['pal'][0] if int(t / TL.BAR) % 2 == 0 else sh['pal'][-1]
    img = np.empty((H, W, 3), np.uint8)
    img[:] = base
    kp = 1 + 0.12 * kick_env(t)
    dt = t - sh['t0']
    for s in MEMPHIS[key]:
        x, y = s['x'] + s['vx'] * dt, s['y'] + s['vy'] * dt
        sz, ang = s['size'] * kp, np.radians(s['rot'] + s['vr'] * dt)
        c = s['col'] if s['col'] != base else WHITE
        p = (int(x), int(y))
        if s['kind'] == 'circle':
            cv2.circle(img, p, int(sz * 0.6), c, -1, cv2.LINE_AA)
            cv2.circle(img, p, int(sz * 0.6), BLACK, 7, cv2.LINE_AA)
        elif s['kind'] == 'ring':
            cv2.circle(img, p, int(sz * 0.7), c, 24, cv2.LINE_AA)
        elif s['kind'] == 'triangle':
            pts = np.array([[x + sz * np.cos(ang + k * 2.094), y + sz * np.sin(ang + k * 2.094)] for k in range(3)], np.int32)
            cv2.fillPoly(img, [pts], c, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, BLACK, 7, cv2.LINE_AA)
        elif s['kind'] == 'zigzag':
            u = np.linspace(-1.2, 1.2, 7) * sz
            v = np.array([(-1) ** k for k in range(7)]) * sz * 0.25
            pts = np.stack([x + u * np.cos(ang) - v * np.sin(ang), y + u * np.sin(ang) + v * np.cos(ang)], 1)
            cv2.polylines(img, [pts.astype(np.int32)], False, c, 18, cv2.LINE_AA)
        elif s['kind'] == 'squiggle':
            u = np.linspace(-1.4, 1.4, 40) * sz
            v = np.sin(u / sz * 6 + dt * 4) * sz * 0.18
            pts = np.stack([x + u * np.cos(ang) - v * np.sin(ang), y + u * np.sin(ang) + v * np.cos(ang)], 1)
            cv2.polylines(img, [pts.astype(np.int32)], False, c, 14, cv2.LINE_AA)
        elif s['kind'] == 'dots':
            for i in range(4):
                for j in range(4):
                    cv2.circle(img, (int(x + (i - 1.5) * sz * 0.4), int(y + (j - 1.5) * sz * 0.4)),
                               int(sz * 0.08), c, -1, cv2.LINE_AA)
        elif s['kind'] == 'cross':
            for a2 in (ang, ang + np.pi / 2):
                d = np.array([np.cos(a2), np.sin(a2)]) * sz * 0.6
                cv2.line(img, (int(x - d[0]), int(y - d[1])), (int(x + d[0]), int(y + d[1])), c, 22, cv2.LINE_AA)
    return img.astype(np.float32) / 255


def bg_marquee(sh, t):
    base, ink = sh['pal']
    img = np.empty((H, W, 3), np.float32)
    img[:] = col(base)
    strip = SPR['marquee'][ink]
    sw = strip.shape[1]
    for i in range(7):
        y = 140 + i * 290
        speed = 260 if i % 2 == 0 else -260
        off = int((speed * t + i * 333) % sw)
        for x in (-off, -off + sw):
            blend(img, strip, x + sw / 2, y, 0.9)
    return img


STAGE = {}


def bg_stage(sh, t):
    if 'base' not in STAGE:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        top, bot = col((38, 12, 62)), col((6, 4, 16))
        base = top * (1 - yy) + bot * yy
        base = np.repeat(base, W, axis=1)
        floor = (np.arange(H) > 1480)[:, None, None]
        base = np.where(floor, col((14, 8, 26)) + 0.08 * (np.linspace(0, 1, H)[:, None, None] - 0.77), base)
        STAGE['base'] = base.astype(np.float32)
    img = STAGE['base'].copy()
    # Lichterwand: Glühbirnen im Takt
    small = np.zeros((H // 4, W // 4, 3), np.float32)
    bi = beat_index(t)
    cols = [PINK, CYAN, YELLOW, PURPLE]
    for r in range(5):
        for c in range(7):
            on = (r + c + bi) % 3 != 0
            cc = col(cols[(r + c) % 4]) * (1.0 if on else 0.15)
            cv2.circle(small, (int((80 + c * 153) / 4), int((150 + r * 150) / 4)), 9, cc.tolist(), -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(small, (0, 0), 6) * 2.2 + small
    img += up(glow) * 0.8
    # Spot auf dem Boden
    yy, xx = np.ogrid[0:H, 0:W]
    spot = np.exp(-(((xx - 540) / 420.0) ** 2 + ((yy - 1600) / 110.0) ** 2))
    img += spot[..., None] * col((255, 230, 190)) * 0.5
    return img


def stage_beams(t):
    """Scheinwerferkegel (additiv, für die Bühne)."""
    LW, LH = 135, 240
    yy, xx = np.mgrid[0:LH, 0:LW].astype(np.float32)
    xx, yy = (xx + 0.5) / LW, (yy + 0.5) / LH
    add = np.zeros((LH, LW, 3), np.float32)
    k = kick_env(t)
    for i, (bx, c) in enumerate(zip((0.05, 0.3, 0.7, 0.95), (PINK, WHITE, WHITE, CYAN))):
        ang = np.radians(16 * np.sin(2 * np.pi * t / (2 * TL.BAR) + i * 1.7) + (15 if bx < 0.5 else -15))
        dx, dy = xx - bx, (yy + 0.03) * H / W
        a = np.arctan2(dx, dy) - ang
        cone = np.exp(-(a / 0.1) ** 2) * np.exp(-dy / 2.5)
        add += cone[..., None] * col(c) * (0.22 + 0.12 * k)
    return up(add)


def background(sh, t):
    return {'sunburst': bg_sunburst, 'tunnel': bg_tunnel, 'memphis': bg_memphis,
            'marquee': bg_marquee, 'stage': bg_stage}[sh['bg']](sh, t)


# ----------------------------------------------------------------- Einstellungen rendern
def shot_at(t):
    for sh in SHOTS:
        if sh['t0'] <= t < sh['t1']:
            return sh
    return SHOTS[-1]


def shot_cam(sh, t):
    return make_cam(sh['img'], sh['fr'], sh['t0'], sh['t1'], t, sh.get('z', (1, 1)), sh.get('rot', (0, 0)))


def render_grid(sh, t, fi):
    tw, th = W // 2, H // 3
    layers = {}
    for k in set(sh['imgs']):
        cam = make_cam(k, 'tile', sh['t0'], sh['t1'], t, (1.0, 1.06))
        layers[k] = (animate(k, cam, (tw, th), fi), cam)
    F = np.zeros((H, W, 3), np.float32)
    for i, k in enumerate(sh['imgs']):
        r, c = divmod(i, 2)
        bgc, ink = GRID_COLORS[(i + beat_index(t)) % len(GRID_COLORS)]
        tile = np.empty((th, tw, 3), np.float32)
        tile[:] = col(bgc)
        lay, cam = layers[k]
        composite_billy(tile, lay, outline=6, shadow=(10, 12), posterize=(BLACK, ink, (255, 245, 225)))
        draw_mic(tile, k, cam, t)
        F[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    F[:, tw - 4:tw + 4] = 0.05
    for r in (1, 2):
        F[r * th - 4:r * th + 4] = 0.05
    return F


def render_shot(sh, t, fi):
    if sh['bg'] == 'grid':
        return render_grid(sh, t, fi)
    F = background(sh, t)
    k = sh['img']
    cam = shot_cam(sh, t)
    lay = animate(k, cam, (W, H), fi)
    if sh['bg'] == 'stage':
        composite_billy(F, lay, outline=0, shadow=None)
    else:
        composite_billy(F, lay)
    draw_mic(F, k, cam, t)
    if sh['bg'] == 'stage':
        b = stage_beams(t)
        F += b * (1 - F)
    return F


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
        ImageDraw.Draw(base).rounded_rectangle((20, 20, bw - 20, bh - 20), radius=38, fill=(14, 6, 24, 175))
        hi = base.copy()
        boxes = {}
        for ri, r in enumerate(rows):
            rw = sum(widths[i] for i in r)
            x = (bw - rw) / 2
            y = 20 + pady + ri * lh
            for i in r:
                for img, c in ((base, WHITE), (hi, YELLOW)):
                    shd = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
                    ImageDraw.Draw(shd).text((x + 2, y + 4), chunks[i], font=fnt, fill=(0, 0, 0, 160))
                    img.alpha_composite(shd.filter(ImageFilter.GaussianBlur(4)))
                    ImageDraw.Draw(img).text((x, y), chunks[i], font=fnt, fill=c + (255,),
                                             stroke_width=4, stroke_fill=(30, 6, 30, 255))
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

BIGWORDS = [(28.0, 'OYE!', YELLOW), (28.5, 'COMO', CYAN), (29.0, 'WAU!', PINK),
            (30.0, 'WAU!', YELLOW), (30.25, 'WAU!', PINK),
            (32.0, 'LORD', YELLOW), (32.5, 'BILLY', PINK), (33.0, 'SINGT', CYAN),
            (33.5, 'FÜR DICH', YELLOW), (34.5, 'AUUU!', PINK)]


BIGWORDS_FREE = [(28.0, 'WAU!', YELLOW), (28.6, 'YEAH!', CYAN), (30.0, 'LORD', PINK), (30.6, 'BILLY', YELLOW),
                 (32.0, 'WAU!', CYAN), (33.0, 'SING!', PINK), (34.5, 'AUUU!', YELLOW)]
if FREE:
    BIGWORDS = BIGWORDS_FREE
    CAPTIONS = []


def credits_sprite():
    lines = [('Lord Billy', font(F_XBOLD, 46)), ('»Oye Como Wau«', font(F_SEMI, 40)),
             ('Löwenmähne', font(F_SEMI, 34)), ('Chow Chow Records', font(F_SEMI, 34))]
    if FREE:
        lines = [('Lord Billy', font(F_XBOLD, 46)), ('Live im WAU-TV-Studio', font(F_SEMI, 40)),
                 ('Chow Chow · 6 Jahre', font(F_SEMI, 34))]
    w = int(max(f.getlength(s) for s, f in lines)) + 60
    hgt = 52 * len(lines) + 50
    img = Image.new('RGBA', (w, hgt), (0, 0, 0, 0))
    shd = Image.new('RGBA', (w, hgt), (0, 0, 0, 0))
    y = 20
    for s, f in lines:
        ImageDraw.Draw(shd).text((22, y + 3), s, font=f, fill=(0, 0, 0, 230), stroke_width=3, stroke_fill=(0, 0, 0, 230))
        y += 52
    img.alpha_composite(shd.filter(ImageFilter.GaussianBlur(4)))
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
           stroke_width=4, stroke_fill=(0, 0, 0, 200))
    d.rounded_rectangle((176, 104, 262, 150), radius=10, fill=PINK + (240,))
    d.text((186, 104), 'TV', font=font(F_BLACK, 40), fill=(255, 255, 255, 255))
    img = img.rotate(8, resample=Image.BICUBIC, expand=True)
    return np.array(img)


def premiere_sprite():
    img = Image.new('RGBA', (420, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, 416, 86), radius=14, fill=PINK + (235,), outline=(0, 0, 0, 255), width=4)
    d.text((24, 14), 'WELTPREMIERE', font=font(F_BLACK, 50), fill=(255, 255, 255, 255))
    return np.array(img)


def marquee_strip(ink):
    fnt = font(F_BLACK, 210)
    txt = 'LORD BILLY • WAU TV • ' if FREE else 'OYE COMO WAU • LORD BILLY • '
    w = int(fnt.getlength(txt)) + 10
    img = Image.new('RGBA', (w, 260), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((0, 0), txt, font=fnt, fill=ink + (120,), stroke_width=6, stroke_fill=ink + (255,))
    return np.array(img)


def build_sprites():
    SPR['mic'] = build_mic()
    SPR['title'] = text_sprite('LORD BILLY', font(F_BLACK, 150), YELLOW, stroke=9, stroke_fill=(20, 0, 30), shadow=14)
    SPR['sub'] = text_sprite('Chow Chow · 6 Jahre', font(F_BOLD, 56), WHITE, stroke=5, stroke_fill=(20, 0, 30))
    SPR['sings'] = text_sprite('singt', font(F_ITAL, 52), WHITE, stroke=5, stroke_fill=(20, 0, 30))
    SPR['song'] = text_sprite('»Oye Como Wau«', font(F_BLACK, 92), PINK, stroke=7, stroke_fill=(20, 0, 30))
    SPR['style'] = text_sprite('Latin-Rock à la Santana', font(F_BOLD, 46), WHITE, stroke=5, stroke_fill=(20, 0, 30))
    SPR['solo'] = text_sprite('GITARRENSOLO!', font(F_BLACK, 112), PINK, stroke=8, stroke_fill=(20, 0, 30))
    SPR['end1'] = text_sprite('LORD BILLY', font(F_BLACK, 140), YELLOW, stroke=9, stroke_fill=(20, 0, 30), shadow=14)
    SPR['end2'] = text_sprite('Danke fürs Zuhören – Wau!', font(F_ITAL, 50), WHITE, stroke=5, stroke_fill=(20, 0, 30))
    if FREE:
        SPR['sings'] = text_sprite('heute live', font(F_ITAL, 52), WHITE, stroke=5, stroke_fill=(20, 0, 30))
        SPR['song'] = text_sprite('»Billy singt!«', font(F_BLACK, 92), PINK, stroke=7, stroke_fill=(20, 0, 30))
        SPR['style'] = text_sprite('Latin-Rock-Special', font(F_BOLD, 46), WHITE, stroke=5, stroke_fill=(20, 0, 30))
        SPR['end2'] = text_sprite('Danke fürs Zuschauen – Wau!', font(F_ITAL, 50), WHITE, stroke=5, stroke_fill=(20, 0, 30))
    SPR['credits'] = credits_sprite()
    SPR['bug'] = bug_sprite()
    SPR['premiere'] = premiere_sprite()
    SPR['marquee'] = {c: marquee_strip(c) for c in (PINK, YELLOW, PURPLE)}
    for i, (_, _, txt) in enumerate(CAPTIONS):
        SPR[f'cap{i}'] = text_sprite(txt, font(F_ITAL, 50), WHITE, stroke=5, stroke_fill=(20, 0, 30), shadow=8)
    SPR['big'] = [text_sprite(w, font(F_BLACK, 170), c, stroke=10, stroke_fill=(20, 0, 30), shadow=12)
                  for _, w, c in BIGWORDS]
    sym = font(F_SYM, 96)
    SPR['notes'] = []
    for g in '♪♫♩♬':
        for c in (YELLOW, PINK, CYAN, WHITE):
            SPR['notes'].append(text_sprite(g, sym, c, stroke=5, stroke_fill=(20, 0, 30), shadow=6))


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
        blend(F, spr, W / 2, 1700 + rise, al)
    for i, (a, b, _) in enumerate(CAPTIONS):
        if a <= t < b:
            blend(F, SPR[f'cap{i}'], W / 2, 1710, window(t, a, b, 0.3))


def draw_bigwords(F, t):
    for i, (t0, _, _) in enumerate(BIGWORDS):
        t1 = BIGWORDS[i + 1][0] if i + 1 < len(BIGWORDS) and BIGWORDS[i + 1][0] - t0 < 1.0 else t0 + 0.9
        if not (t0 <= t < t1):
            continue
        a = t - t0
        sc = 1.0 + 0.55 * np.exp(-a / 0.06)
        ang = 6 if i % 2 == 0 else -6
        blend(F, xform(SPR['big'][i], sc, ang), W / 2, 250, smooth(a / 0.04) * smooth((t1 - t) / 0.1))


def draw_titles(F, t):
    if t < 4.2:
        a = window(t, 0.25, 3.85, 0.35)
        sc = 0.85 + 0.15 * smooth((t - 0.25) / 0.5) + 0.04 * kick_env(t)
        blend(F, xform(SPR['title'], sc), W / 2, 270, a)
        blend(F, SPR['sub'], W / 2, 400, window(t, 0.7, 3.85, 0.35))
        blend(F, SPR['sings'], W / 2, 1500, window(t, 1.5, 3.85, 0.3))
        blend(F, xform(SPR['song'], 0.9 + 0.1 * smooth((t - 1.8) / 0.4)), W / 2, 1595, window(t, 1.8, 3.85, 0.3))
        blend(F, SPR['style'], W / 2, 1700, window(t, 2.3, 3.85, 0.3))
        blink = 1.0 if int(t * 2.5) % 2 == 0 else 0.55
        blend(F, SPR['premiere'], 60 + SPR['premiere'].shape[1] / 2, 110, window(t, 0.3, 3.9, 0.2) * blink)
    if 0.6 <= t < 44.6:
        b = SPR['bug']
        blend(F, b, W - 40 - b.shape[1] / 2, 50 + b.shape[0] / 2, 0.85 * window(t, 0.6, 44.6, 0.4))
    c = SPR['credits']
    for a, bnd in ((4.4, 9.6), (42.4, 44.8)):
        if a <= t < bnd:
            blend(F, c, 40 + c.shape[1] / 2, 1380 + c.shape[0] / 2, window(t, a, bnd, 0.35))
    for a, b in (() if FREE else ((36.0, 39.0), (40.0, 42.0))):
        if a <= t < b:
            sc = 1.0 + 0.1 * kick_env(t)
            blend(F, xform(SPR['solo'], sc, 4 * np.sin(t * 6)), W / 2, 300, window(t, a, b, 0.15))
    if t >= 42.5:
        blend(F, SPR['end1'], W / 2, 300, window(t, 42.5, 50, 0.5))
        blend(F, SPR['end2'], W / 2, 430, window(t, 43.0, 50, 0.5))


# ----------------------------------------------------------------- Noten-Partikel
PARTICLES = []


def build_particles():
    rng = np.random.default_rng(6)
    for n in notes_list():
        cnt = 1 + int(n['t1'] - n['t0'] > 0.6) + int(n['t1'] - n['t0'] > 1.2)
        for j in range(cnt):
            ts = n['t0'] + j * 0.35
            sh = shot_at(ts)
            if sh['bg'] == 'grid':
                xc, y, half = W / 2, H / 2, 200
            else:
                g = mouth_geom(sh['img'], shot_cam(sh, ts))
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
    load_cutouts()
    build_sprites()
    if not FREE:
        build_lyrics()
    build_particles()
    MOUTH = free_mouth(np.arange(NF) / FPS) if FREE else TL.mouth_open(np.arange(NF) / FPS)
    compute_pose()


def vhs(F, t, t_cut, fi):
    """Kurzes VHS-Rauschen mit Bildversatz direkt nach einem Schnitt."""
    a = t - t_cut
    if not (0 <= a < 0.12):
        return F
    rng = np.random.default_rng(fi)
    k = 1 - a / 0.12
    noise = rng.random((H // 4, W // 4)).astype(np.float32)
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
    out = F * (1 - 0.55 * k) + noise * 0.55 * k
    for _ in range(6):   # horizontale Risse
        y = int(rng.integers(0, H - 60))
        h = int(rng.integers(10, 60))
        out[y:y + h] = np.roll(out[y:y + h], int(rng.integers(-80, 80)), axis=1)
    return out


def render(fi):
    t = fi / FPS
    sh = shot_at(t)
    sec = TL.section_at(t)
    F = render_shot(sh, t, fi)
    # Stroboskop / Blitze
    if sec == 'chorus' and not FREE:
        snares = np.array([TL.bt(b, k) for b in range(14, 18) for k in (1, 3)])
        F += (1 - F) * 0.2 * env_after(snares, t, 0.05)
    if sec == 'solo' and not FREE:
        F += (1 - F) * 0.15 * env_after(BEATS, t, 0.05)
    for s in SHOTS:
        if s['flash'] and 0 <= t - s['t0'] < 0.25:
            F += (1 - F) * 0.5 * np.exp(-(t - s['t0']) / 0.06)
        if s.get('vhs'):
            F = vhs(F, t, s['t0'], fi)
    # RGB-Versatz auf Kicks und Schnitten
    cut = max([np.exp(-(t - s['t0']) / 0.1) for s in SHOTS if 0 <= t - s['t0'] < 0.4] + [0])
    d = int(round((12 if sec in ('chorus', 'solo') else 5) * kick_env(t) + 10 * cut))
    if d > 0:
        F[:, d:, 0] = F[:, :-d, 0].copy()
        F[:, :-d, 2] = F[:, d:, 2].copy()
    np.clip(F, 0, 1, out=F)
    draw_particles(F, t)
    draw_bigwords(F, t)
    draw_titles(F, t)
    draw_lyrics(F, t)
    F *= smooth(t / 0.4) * smooth((TL.DURATION - t) / 0.8)
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
    if sys.argv[1] == '--ohne-ton':
        wav, mp4 = None, sys.argv[2]
    else:
        wav, mp4 = sys.argv[1], sys.argv[2]
    init_worker()
    if not FREE:
        write_srt(os.path.splitext(mp4)[0] + '.srt')
    build = os.path.join(ROOT, 'build')
    os.makedirs(build, exist_ok=True)
    raw = os.path.join(build, 'video_raw.mp4')
    # 1) Bilder in 1080x1920 rendern, als hochwertiges Zwischenvideo in 720x1280 ablegen
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-vf', f'scale={OUT_W}:{OUT_H}:flags=lanczos',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '12', '-pix_fmt', 'yuv420p', raw]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker) as pool:
        for i, fr in enumerate(pool.imap(render, range(NF), chunksize=4)):
            proc.stdin.write(fr.tobytes())
            if i % 150 == 0:
                print(f'  Bild {i}/{NF}', flush=True)
    proc.stdin.close()
    proc.wait()
    # 2) Zwei-Pass-Encoding auf eine feste Zielgröße (unter 20 MB)
    vbit = int(TARGET_MB * 8e6 / TL.DURATION - (0 if FREE else 128e3))
    log = os.path.join(build, 'x264pass')
    common = ['-c:v', 'libx264', '-preset', 'slow', '-b:v', str(vbit), '-maxrate', str(int(vbit * 1.6)),
              '-bufsize', str(int(vbit * 2)), '-pix_fmt', 'yuv420p', '-passlogfile', log]
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw] + common + ['-pass', '1', '-an', '-f', 'mp4', os.devnull],
                   check=True)
    audio = ['-an'] if FREE else ['-c:a', 'aac', '-b:a', '128k', '-shortest']
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw] + ([] if FREE else ['-i', wav]) + common +
                   ['-pass', '2'] + audio + ['-movflags', '+faststart', mp4], check=True)
    print('fertig:', mp4, f'{os.path.getsize(mp4) / 1e6:.1f} MB')


if __name__ == '__main__':
    if '--ohne-ton' in sys.argv and not FREE:   # Modus vor dem Start der Worker festlegen
        os.environ['LB_OHNE_TON'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
