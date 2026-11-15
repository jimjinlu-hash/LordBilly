"""Rendert das Musikvideo "Lord Billy – Oye Como Wau" (1080x1920, 30 fps).

Aus den drei Fotos entstehen Kamerafahrten; Billys Unterkiefer wird synchron
zum Gesang per Verzerrungsfeld geöffnet. Dazu Karaoke-Untertitel der gerade
gesungenen Zeile, fliegende Noten, Lichteffekte und Titel.

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
F_ITAL = FONT_DIR + 'InterDisplay-SemiBoldItalic.otf'
F_SYM = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

GOLD = (255, 200, 61)
ORANGE = (255, 128, 40)
PINK = (255, 92, 150)
WHITE = (255, 255, 255)

# Maul-Geometrie je Foto (Pixel im Originalbild): Mitte x, Oberlippe y,
# Höhe des dunklen Spalts, Unterkiefer-Unterkante, Ausklingzone darunter,
# halbe Breite, Neigung der Lippenlinie, maximale Öffnung.
IMAGES = {
    'A': dict(file='bilder/d31ad93b38b30e8aafbaaa2372b6c3cc.jpeg', prescale=0.75,
              mouth=dict(x=1470, lip=1345, gap=70, jaw=1700, decay=420, half=330, slope=0.0, amax=115)),
    'B': dict(file='bilder/ed453128c9dc6df5e54131ae1af421_livephoto.jpeg', prescale=1.0,
              mouth=dict(x=965, lip=842, gap=42, jaw=975, decay=230, half=180, slope=0.06, amax=58)),
    'C': dict(file='bilder/IMG_1332.jpeg', prescale=1.0, sharpen=True,
              mouth=dict(x=521, lip=627, gap=9, jaw=683, decay=95, half=52, slope=0.0, amax=19)),
}

# Einstellungen: (Start, Ende, Foto, Kamera Anfang, Kamera Ende, Blitz beim Schnitt)
# Kamera = (Mitte x, Mitte y, Quellpixel je Ausgabepixel, Drehung in Grad)
SHOTS = [
    (0.0, 4.0, 'C', (540, 545, 0.54, 0), (528, 600, 0.47, 0), False),
    (4.0, 12.0, 'B', (960, 1280, 1.30, 0), (915, 900, 0.95, 0), False),
    (12.0, 20.0, 'A', (1470, 1700, 1.75, 0), (1460, 1420, 1.35, 0), True),
    (20.0, 28.0, 'B', (930, 960, 1.0, 0), (950, 860, 0.8, 0), True),
    (28.0, 30.0, 'A', (1460, 1450, 1.3, 0), (1460, 1430, 1.15, 0), True),
    (30.0, 32.0, 'C', (525, 640, 0.44, 0), (522, 640, 0.40, 0), True),
    (32.0, 34.0, 'B', (950, 870, 0.85, 0), (940, 870, 0.75, 0), True),
    (34.0, 36.0, 'A', (1460, 1450, 1.2, 0), (1460, 1400, 1.05, -2.0), True),
    (36.0, 37.0, 'C', (528, 600, 0.48, 0), (528, 600, 0.44, 0), True),
    (37.0, 38.0, 'B', (920, 820, 0.95, 0), (920, 820, 0.85, 0), True),
    (38.0, 39.0, 'C', (525, 640, 0.42, 0), (525, 640, 0.38, 0), True),
    (39.0, 40.0, 'A', (1460, 1450, 1.15, 0), (1460, 1420, 1.0, -2.0), True),
    (40.0, 41.0, 'B', (950, 870, 0.8, 0), (950, 870, 0.7, 0), True),
    (41.0, 42.0, 'C', (540, 545, 0.54, 0), (530, 560, 0.5, 0), True),
    (42.0, 45.0, 'A', (1460, 1430, 1.1, -1.5), (1480, 1920, 2.0, 0), True),
]
CROSSFADES = {4.0: 0.45}


# ----------------------------------------------------------------- Hilfen
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease(x):
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def kick_env(t):
    ks = TL.kick_times()
    past = ks[ks <= t]
    if len(past) == 0:
        return 0.0
    return float(np.exp(-(t - past[-1]) / 0.13))


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
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32) / 255.0
    a = s[..., 3:4] * alpha
    d = dst[ya:yb, xa:xb]
    d *= (1 - a)
    d += s[..., :3] * a


# ----------------------------------------------------------------- Bilder
_IMG = {}


def load_images():
    for k, spec in IMAGES.items():
        im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, spec['file']))).convert('RGB')
        if spec.get('sharpen'):
            im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
        a = np.asarray(im).astype(np.float32)
        # warmer Look: etwas mehr Rot/Gold, weniger Blau, sanfter Kontrast
        a[..., 0] *= 1.05
        a[..., 2] *= 0.93
        a = 255 * (0.5 + 1.08 * (a / 255 - 0.5))
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


def camera(shot, t):
    t0, t1, k, c0, c1, _ = shot
    u = ease((t - t0) / (t1 - t0))
    cx, cy, s, rot = [a + (b - a) * u for a, b in zip(c0, c1)]
    sec = TL.section_at(t)
    pulse = 0.03 if sec in ('chorus', 'solo') else 0.016
    s *= 1 - pulse * kick_env(t)
    sway = 1.4 if sec == 'solo' else 0.8
    rot += sway * np.sin(2 * np.pi * t / (2 * TL.BAR))
    return clamp_cam(k, cx, cy, s, rot)


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
    # R(-th) v
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


def shot_at(t):
    for sh in SHOTS:
        if sh[0] <= t < sh[1]:
            return sh
    return SHOTS[-1]


def photo_frame(t, mouth):
    sh = shot_at(t)
    out = None
    for cut, dur in CROSSFADES.items():
        if cut - dur / 2 <= t < cut + dur / 2:
            prev = shot_at(cut - 1e-3)
            nxt = shot_at(cut)
            w = smooth((t - (cut - dur / 2)) / dur)
            fa = render_layer(prev, min(t, prev[1] - 1e-3), mouth)
            fb = render_layer(nxt, max(t, nxt[0]), mouth)
            out = (fa.astype(np.float32) * (1 - w) + fb.astype(np.float32) * w).astype(np.uint8)
            sh = nxt if w > 0.5 else prev
    if out is None:
        out = render_layer(sh, t, mouth)
    return out, sh


def render_layer(sh, t, mouth):
    cam = camera(sh, t)
    fr = render_photo(sh[2], cam)
    return open_mouth(fr, mouth_geom(sh[2], cam), mouth)


# ----------------------------------------------------------------- Licht & Look
LW, LH = 135, 240
_yy, _xx = np.mgrid[0:LH, 0:LW].astype(np.float32)
_xx = (_xx + 0.5) / LW
_yy = (_yy + 0.5) / LH


def light_maps(t):
    k = kick_env(t)
    r = np.sqrt(((_xx - 0.5) * 1.1) ** 2 + ((_yy - 0.47) * 0.75) ** 2)
    vig = 1 - (0.55 + 0.1 * k) * smooth((r - 0.22) / 0.42)
    bottom = 1 - 0.5 * smooth((_yy - 0.7) / 0.28)       # Lesbarkeit der Untertitel
    mult = vig * bottom
    add = np.zeros((LH, LW, 3), np.float32)
    blobs = [((1.0, 0.55, 0.15), 0.25, 0.10, 0.11, 0.0), ((1.0, 0.25, 0.5), 0.8, 0.18, 0.07, 2.0),
             ((1.0, 0.8, 0.3), 0.6, 0.85, 0.09, 4.0)]
    for col, bx, by, sp, ph in blobs:
        x = bx + 0.25 * np.sin(sp * 2 * np.pi * t + ph)
        y = by + 0.08 * np.cos(sp * 1.7 * np.pi * t + ph)
        d2 = ((_xx - x) * 1.0) ** 2 + ((_yy - y) * 0.56) ** 2
        add += np.exp(-d2 / 0.03)[..., None] * np.array(col, np.float32) * (0.16 + 0.08 * k)
    mult = cv2.resize(mult, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    add = cv2.resize(add, (W, H), interpolation=cv2.INTER_LINEAR)
    return mult, add


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
        # Zeilenumbruch an Wortgrenzen (Fallback, falls zu breit)
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
        ImageDraw.Draw(base).rounded_rectangle((20, 20, bw - 20, bh - 20), radius=38, fill=(20, 8, 4, 120))
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
    # Sichtbarkeitsfenster ohne Überlappung
    for i, L in enumerate(LYR):
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


def build_sprites():
    SPR['title'] = text_sprite('LORD BILLY', font(F_BLACK, 150), GOLD, stroke=8, shadow=14)
    SPR['sub'] = text_sprite('Chow Chow · 6 Jahre', font(F_BOLD, 56), WHITE, stroke=4)
    SPR['sings'] = text_sprite('singt', font(F_ITAL, 52), WHITE, stroke=4)
    SPR['song'] = text_sprite('»Oye Como Wau«', font(F_BLACK, 92), ORANGE, stroke=6, stroke_fill=(50, 10, 0))
    SPR['style'] = text_sprite('Latin-Rock à la Santana', font(F_BOLD, 46), WHITE, stroke=4)
    note = text_sprite('♪', font(F_SYM, 36), GOLD, stroke=3, shadow=6)
    txt = text_sprite('Lord Billy – Oye Como Wau', font(F_BOLD, 36), WHITE, stroke=3, shadow=6)
    h = max(note.shape[0], txt.shape[0])
    badge = np.zeros((h, note.shape[1] + txt.shape[1] - 20, 4), np.uint8)
    badge[(h - note.shape[0]) // 2:(h - note.shape[0]) // 2 + note.shape[0], :note.shape[1]] = note
    sub = badge[(h - txt.shape[0]) // 2:(h - txt.shape[0]) // 2 + txt.shape[0], note.shape[1] - 20:]
    np.maximum(sub, txt, out=sub)
    SPR['badge'] = badge
    SPR['solo'] = text_sprite('GITARRENSOLO!', font(F_BLACK, 112), PINK, stroke=7, stroke_fill=(40, 0, 20))
    SPR['end1'] = text_sprite('LORD BILLY', font(F_BLACK, 140), GOLD, stroke=8, shadow=14)
    SPR['end2'] = text_sprite('Oye Como Wau · 2026', font(F_BOLD, 54), WHITE, stroke=4)
    SPR['end3'] = text_sprite('Danke fürs Zuhören – Wau!', font(F_ITAL, 50), WHITE, stroke=4)
    for i, (_, _, txt) in enumerate(CAPTIONS):
        SPR[f'cap{i}'] = text_sprite(txt, font(F_ITAL, 50), WHITE, stroke=4, shadow=8)
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
        blend(F, spr, W / 2, 1640 + rise, al)
    for i, (a, b, _) in enumerate(CAPTIONS):
        if a <= t < b:
            blend(F, SPR[f'cap{i}'], W / 2, 1650, window(t, a, b, 0.3))


def scaled(spr, sc):
    if abs(sc - 1) < 0.01:
        return spr
    return cv2.resize(spr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)


def draw_titles(F, t):
    # Intro
    if t < 4.2:
        a = window(t, 0.25, 3.85, 0.35)
        sc = 0.85 + 0.15 * smooth((t - 0.25) / 0.5) + 0.04 * kick_env(t)
        blend(F, scaled(SPR['title'], sc), W / 2, 250, a)
        blend(F, SPR['sub'], W / 2, 385, window(t, 0.7, 3.85, 0.35))
        blend(F, SPR['sings'], W / 2, 1450, window(t, 1.5, 3.85, 0.3))
        blend(F, scaled(SPR['song'], 0.9 + 0.1 * smooth((t - 1.8) / 0.4)), W / 2, 1545, window(t, 1.8, 3.85, 0.3))
        blend(F, SPR['style'], W / 2, 1650, window(t, 2.3, 3.85, 0.3))
    # Laufendes Badge
    if 4.3 <= t < 42.2:
        blend(F, SPR['badge'], W / 2, 92, 0.85 * window(t, 4.3, 42.2, 0.5))
    # Solo
    for a, b in ((36.0, 39.0), (40.0, 42.0)):
        if a <= t < b:
            sc = 1.0 + 0.08 * kick_env(t)
            blend(F, scaled(SPR['solo'], sc), W / 2, 270, window(t, a, b, 0.2))
    # Abspann
    if t >= 42.6:
        blend(F, SPR['end1'], W / 2, 300, window(t, 42.6, 50, 0.5))
        blend(F, SPR['end2'], W / 2, 430, window(t, 43.0, 50, 0.5))
        blend(F, SPR['end3'], W / 2, 1460, window(t, 43.5, 50, 0.5))


# ----------------------------------------------------------------- Noten-Partikel
PARTICLES = []


def build_particles():
    rng = np.random.default_rng(6)
    for n in TL.vocal_notes():
        cnt = 1 + int(n['t1'] - n['t0'] > 0.6) + int(n['t1'] - n['t0'] > 1.2)
        for j in range(cnt):
            ts = n['t0'] + j * 0.35
            sh = shot_at(ts)
            g = mouth_geom(sh[2], camera(sh, ts))
            side = rng.choice([-1, 1])
            PARTICLES.append(dict(
                t0=ts, life=rng.uniform(1.5, 2.1), spr=int(rng.integers(len(SPR['notes']))),
                x=g['xc'] + side * g['half'] * rng.uniform(0.6, 1.0), y=g['lip'] + g['J'] * 0.35,
                vx=side * rng.uniform(70, 170), vy=-rng.uniform(260, 380),
                sc=rng.uniform(0.6, 1.05), ph=rng.uniform(0, 6.28)))


def draw_particles(F, t):
    for p in PARTICLES:
        a = t - p['t0']
        if a < 0 or a > p['life']:
            continue
        x = p['x'] + p['vx'] * a + 22 * np.sin(4 * a + p['ph'])
        y = p['y'] + p['vy'] * a + 40 * a * a
        al = smooth(a / 0.12) * smooth((p['life'] - a) / 0.5)
        sc = p['sc'] * (0.7 + 0.3 * smooth(a / 0.3))
        blend(F, scaled(SPR['notes'][p['spr']], sc), x, y, al)


# ----------------------------------------------------------------- Bild komponieren
MOUTH = None


def init_worker():
    global MOUTH
    load_images()
    build_sprites()
    build_lyrics()
    build_particles()
    MOUTH = TL.mouth_open(np.arange(NF) / FPS)


def render(fi):
    t = fi / FPS
    photo, sh = photo_frame(t, float(MOUTH[fi]))
    F = photo.astype(np.float32) / 255.0
    mult, add = light_maps(t)
    F *= mult
    F += add * (1 - F)
    draw_particles(F, t)
    draw_titles(F, t)
    draw_lyrics(F, t)
    # Blitz bei harten Schnitten
    for s in SHOTS:
        if s[5] and 0 <= t - s[0] < 0.25:
            F += (1 - F) * 0.55 * np.exp(-(t - s[0]) / 0.07)
    # Auf- und Abblende
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
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p',
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
