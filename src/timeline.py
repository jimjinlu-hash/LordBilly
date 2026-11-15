"""Gemeinsamer Zeitplan für Song und Video: "Oye Como Wau" von Lord Billy.

Eigenkomposition im Latin-Rock-Stil (Am7–D9-Vamp, Cha-Cha-Groove, singende
Leadgitarre) als Hommage an Santana. Text und Melodie sind original.

Der Gesang wird als Phonemfolge für MBROLA (Stimme de6, deutsch, männlich)
geplant. Aus derselben Phonemfolge berechnet das Video die Maulbewegung.
"""
import numpy as np

BPM = 120
BEAT = 60.0 / BPM          # 0,5 s
BAR = 4 * BEAT             # 2,0 s
DURATION = 45.0
SR = 44100
FPS = 30
VOICE_TRANSPOSE = -24      # zwei Oktaven tiefer: tiefe Männerstimme (A2–A3)


def bt(bar, beat=0.0):
    """Zeit in Sekunden für Takt `bar` (ab 0) und Schlag `beat` (ab 0)."""
    return bar * BAR + beat * BEAT


A4, B4, C5, D5, E5, Fs5, G5, A5, C6 = 69, 71, 72, 74, 76, 78, 79, 81, 84
G4 = 67

# --- Gesang ------------------------------------------------------------------
# (Text-Stück, Schlag, Dauer in Schlägen, MIDI, Phoneme "Anlaut|Vokal|Auslaut", Extras)
# Die Text-Stücke ergeben aneinandergehängt die angezeigte Liedzeile.
LINES = [
    dict(bar=5, part='warmup', syl=[
        ("Au", 1.0, 0.5, D5, "?|aU|", {}),
        ("-uuu?", 1.5, 1.75, A5, "|u:|", dict(scoop=-3, glide_to=G5)),
    ]),
    dict(bar=6, part='verse', syl=[
        ("O", 0.0, 0.5, E5, "?|o:|", {}),
        ("ye, ", 0.5, 1.0, D5, "j|e:|", {}),
        ("mein ", 1.5, 0.5, C5, "m|aI|n", {}),
        ("Na", 2.0, 0.5, D5, "n|a:|", {}),
        ("me ", 2.5, 0.5, C5, "m|@|", {}),
        ("ist ", 3.0, 0.5, A4, "?|I|s t", {}),
        ("Bil", 3.5, 1.0, C5, "b|I|l", {}),
        ("ly", 4.5, 2.0, A4, "|i:|", {}),
    ]),
    dict(bar=8, part='verse', syl=[
        ("ein ", 0.0, 0.5, A4, "?|aI|n", {}),
        ("Chow ", 0.5, 0.5, C5, "tS|aU|", {}),
        ("Chow ", 1.0, 1.0, D5, "tS|aU|", {}),
        ("mit ", 2.0, 0.5, E5, "m|I|t", {}),
        ("Lö", 2.5, 0.5, G5, "l|2:|", {}),
        ("wen", 3.0, 0.5, E5, "v|@|n", {}),
        ("mäh", 3.5, 1.0, D5, "m|E:|", {}),
        ("ne", 4.5, 2.0, E5, "n|E|", {}),
    ]),
    dict(bar=10, part='verse', syl=[
        ("Sechs ", 0.0, 0.5, E5, "z|E|k s", {}),
        ("Jah", 0.5, 0.5, E5, "j|a:|", {}),
        ("re ", 1.0, 1.0, D5, "R|@|", {}),
        ("Son", 2.0, 0.5, C5, "z|O|n", {}),
        ("ne ", 2.5, 0.5, D5, "n|@|", {}),
        ("im ", 3.0, 0.5, E5, "?|I|m", {}),
        ("Fell", 3.5, 3.0, G5, "f|E|l", dict(glide_to=E5)),
    ]),
    dict(bar=12, part='verse', syl=[
        ("die ", 0.0, 0.5, A4, "d|i:|", {}),
        ("Zun", 0.5, 0.5, C5, "ts|U|N", {}),
        ("ge ", 1.0, 0.5, D5, "g|@|", {}),
        ("so ", 1.5, 0.5, E5, "z|o:|", {}),
        ("blau ", 2.0, 1.0, G5, "b l|aU|", {}),
        ("wie ", 3.0, 0.5, E5, "v|i:|", {}),
        ("die ", 3.5, 0.5, D5, "d|i:|", {}),
        ("Nacht", 4.0, 3.0, A4, "n|a|x t", {}),
    ]),
    dict(bar=14, part='chorus', syl=[
        ("O", 0.0, 0.5, E5, "?|o:|", {}),
        ("ye ", 0.5, 0.5, G5, "j|e:|", {}),
        ("co", 1.0, 0.5, E5, "k|o:|", {}),
        ("mo ", 1.5, 0.5, D5, "m|o:|", {}),
        ("wau, ", 2.0, 1.5, E5, "v|aU|", {}),
        ("wau-", 4.0, 0.5, D5, "v|aU|", {}),
        ("wau!", 4.5, 1.75, C5, "v|aU|", dict(glide_to=A4)),
    ]),
    dict(bar=16, part='chorus', syl=[
        ("Lord ", 0.0, 1.0, A4, "l|O|6 t", {}),
        ("Bil", 1.0, 0.5, C5, "b|I|l", {}),
        ("ly ", 1.5, 0.5, D5, "|i:|", {}),
        ("singt ", 2.0, 0.5, E5, "z|I|N t", {}),
        ("für ", 2.5, 0.5, G5, "f|y:|6", {}),
        ("dich – ", 3.0, 1.0, A5, "d|I|C", {}),
        ("au", 4.5, 0.5, E5, "?|aU|", {}),
        ("uuu!", 5.0, 2.5, A5, "|u:|", dict(scoop=-3)),
    ]),
    dict(bar=19, part='solo', syl=[
        ("Au", 2.0, 0.5, E5, "?|aU|", {}),
        ("-uuuu!", 2.5, 1.5, A5, "|u:|", dict(scoop=-3)),
    ]),
    dict(bar=21, part='finale', syl=[
        ("Wau", 0.0, 1.0, A4, "v|aU|", dict(glide_to=E5)),
        ("-uuuuuh!", 1.0, 3.6, A5, "|u:|", dict(scoop=-3, glide_to=E5, fade=True)),
    ]),
]


def vocal_notes():
    """Liste aller gesungenen Silben mit absoluten Zeiten."""
    notes = []
    for li, line in enumerate(LINES):
        for si, (txt, beat, dur, midi, phon, extra) in enumerate(line['syl']):
            on, vow, cod = phon.split('|')
            t0 = bt(line['bar'], beat)
            notes.append(dict(line=li, idx=si, text=txt, t0=t0, t1=t0 + dur * BEAT,
                              midi=midi, onset=on.split(), vowel=vow, coda=cod.split(), **extra))
    return notes


def line_spans():
    """(Start, Ende) jeder Liedzeile für die Karaoke-Anzeige."""
    spans = []
    notes = vocal_notes()
    for li in range(len(LINES)):
        ns = [n for n in notes if n['line'] == li]
        spans.append((ns[0]['t0'], ns[-1]['t1']))
    return spans


# --- Phoneme ------------------------------------------------------------------
CONS_DUR = {'?': 0.035, 'p': 0.06, 't': 0.06, 'k': 0.065, 'b': 0.055, 'd': 0.055,
            'g': 0.055, 'f': 0.08, 'v': 0.065, 's': 0.085, 'z': 0.075, 'S': 0.09,
            'C': 0.085, 'x': 0.085, 'j': 0.06, 'h': 0.06, 'm': 0.07, 'n': 0.065,
            'N': 0.065, 'l': 0.065, 'R': 0.045, '6': 0.07, 'ts': 0.09, 'tS': 0.1}
MOUTH = {'a': 1.0, 'a:': 1.0, 'E': 0.75, 'E:': 0.8, 'e:': 0.65, '@': 0.5, '6': 0.55,
         'I': 0.45, 'i:': 0.42, 'O': 0.8, 'o:': 0.75, 'U': 0.55, 'u:': 0.5, '2:': 0.5,
         'y:': 0.45, 'Y': 0.45, '9': 0.6,
         'aU': (1.0, 0.55), 'aI': (1.0, 0.45), 'OY': (0.8, 0.45),
         'm': 0.0, 'b': 0.0, 'p': 0.0, 'f': 0.15, 'v': 0.15, '?': 0.35, '_': 0.0}


def note_pitch(n, t):
    """MIDI-Tonhöhe (transponiert) einer Silbe zur Zeit t (Array, relativ zum Vokalbeginn)."""
    d = n['t1'] - n['t0']
    m = np.full_like(t, float(n['midi']), dtype=float)
    scoop = n.get('scoop', -0.8)
    gl = 0.08 if abs(scoop) < 2 else 0.18
    m += scoop * (1 - np.clip(t / gl, 0, 1)) ** 2
    if n.get('glide_to') is not None:
        x = np.clip((t - 0.6 * d) / (0.4 * d + 0.1), 0, 1)
        m += (n['glide_to'] - n['midi']) * x * x * (3 - 2 * x)
    if d > 0.3:
        ramp = np.clip((t - 0.2) / 0.3, 0, 1)
        m += 0.3 * ramp * np.sin(2 * np.pi * 5.3 * t)
    return m + VOICE_TRANSPOSE


def phonemes():
    """Phonemfolge mit absoluten Zeiten: Liste von dict(ph, t0, t1, note, kind)."""
    notes = vocal_notes()
    segs = []
    cur = 0.0

    def push(ph, a, b, note=None, kind='c'):
        nonlocal cur
        if b - a > 0.004:
            segs.append(dict(ph=ph, t0=a, t1=b, note=note, kind=kind))
            cur = b

    for i, n in enumerate(notes):
        on = sum(CONS_DUR.get(c, 0.06) for c in n['onset'])
        start = n['t0'] - on
        if start > cur + 0.004:
            push('_', cur, start)
        else:
            start = cur
        # Ende dieser Silbe: Legato bis zum Anlaut der nächsten, sonst Notenende
        E = n['t1']
        if i + 1 < len(notes):
            nx = notes[i + 1]
            non = sum(CONS_DUR.get(c, 0.06) for c in nx['onset'])
            if nx['t0'] - n['t1'] < non + 0.03:
                E = nx['t0'] - non
        cod = [CONS_DUR.get(c, 0.06) for c in n['coda']]
        lc = sum(cod)
        v0 = max(start + on * 0.5, n['t0']) if on else start
        if E - lc - v0 < 0.06 and lc > 0:
            k = max(E - v0 - 0.06, 0.02) / lc
            cod = [c * k for c in cod]
            lc = sum(cod)
        # Anlaut
        t = start
        span_on = v0 - start
        for c in n['onset']:
            d = CONS_DUR.get(c, 0.06) * (span_on / on if on else 0)
            push(c, t, t + d, n, 'on')
            t += d
        push(n['vowel'], t, E - lc, n, 'v')
        t = E - lc
        for c, d in zip(n['coda'], cod):
            push(c, t, t + d, n, 'cod')
            t += d
    push('_', cur, DURATION + 0.5)
    return segs


def pho_file(segs, pitch_factor=1.0, transpose=0.0):
    """MBROLA-.pho-Text mit Tonhöhenpunkten (Vibrato, Scoops, Glides)."""
    lines = []
    acc = 0
    for s in segs:
        a, b = round(s['t0'] * 1000), round(s['t1'] * 1000)
        d = b - a
        if d <= 0:
            continue
        acc += d
        n = s['note']
        if n is None:
            lines.append(f"_ {d}")
            continue
        vstart = next(x['t0'] for x in segs if x['note'] is n and x['kind'] == 'v')
        if s['kind'] == 'v':
            pcts = np.linspace(0, 100, max(2, min(40, int(d / 25) + 2)))
        else:
            pcts = np.array([0.0, 100.0])
        ts = s['t0'] + pcts / 100 * (s['t1'] - s['t0']) - vstart
        ts = np.clip(ts, 0, None) if s['kind'] != 'cod' else ts
        hz = 440 * 2 ** ((note_pitch(n, ts) + transpose - 69) / 12) * pitch_factor
        pts = ' '.join(f"{p:.0f} {h:.1f}" for p, h in zip(pcts, hz))
        lines.append(f"{s['ph']} {d} {pts}")
    return '\n'.join(lines) + '\n'


def mouth_open(t):
    """Maulöffnung 0..1 zum Zeitpunkt t (Array), aus der Phonemfolge."""
    from scipy import signal
    grid = np.arange(0, DURATION + 0.5, 0.001)
    val = np.zeros_like(grid)
    for s in phonemes():
        i0, i1 = int(s['t0'] * 1000), int(s['t1'] * 1000)
        if i1 <= i0:
            continue
        m = MOUTH.get(s['ph'], 0.3)
        if isinstance(m, tuple):
            x = np.linspace(0, 1, i1 - i0)
            val[i0:i1] = m[0] + (m[1] - m[0]) * np.clip((x - 0.35) / 0.5, 0, 1)
        else:
            val[i0:i1] = m
        n = s['note']
        if n is not None and n.get('fade') and s['kind'] == 'v':
            x = np.linspace(0, 1, i1 - i0)
            val[i0:i1] *= np.clip(1.15 - x, 0.3, 1)
    val = signal.sosfiltfilt(signal.butter(2, 9, fs=1000, output='sos'), val)
    return np.clip(np.interp(np.asarray(t, float), grid, val), 0, 1.05)


# --- Leadgitarre ----------------------------------------------------------------
GUITAR = []


def _g(bar, items, gain=1.0):
    for beat, dur, midi, *rest in items:
        bend = rest[0] if rest else 0
        GUITAR.append(dict(t0=bt(bar, beat), t1=bt(bar, beat + dur), midi=midi,
                           bend=bend, gain=gain))


# Intro-Riff (Takt 2–5)
_g(2, [(0.0, 1.5, E5, -2), (1.5, 0.5, D5), (2.0, 0.5, C5), (2.5, 1.5, A4)])
_g(3, [(0.0, 0.5, G4), (0.5, 0.5, A4), (1.0, 1.0, C5), (2.0, 2.0, D5, -2)])
_g(4, [(0.0, 1.0, A5, -2), (1.0, 0.5, G5), (1.5, 0.5, E5), (2.0, 0.5, D5),
       (2.5, 0.5, E5), (3.0, 1.0, C5)])
_g(5, [(0.0, 1.0, A4)])
_g(5, [(3.0, 0.5, E5), (3.5, 0.5, G5)], gain=0.8)
# kleine Fills zwischen den Gesangszeilen
_g(7, [(2.5, 0.5, A5), (3.0, 0.5, G5), (3.5, 0.5, E5)], gain=0.55)
_g(9, [(2.5, 0.5, E5), (3.0, 0.5, G5), (3.5, 0.5, A5, -2)], gain=0.55)
_g(11, [(3.0, 0.5, D5), (3.5, 0.5, C5)], gain=0.5)
_g(13, [(3.0, 0.5, E5), (3.5, 0.5, G5)], gain=0.6)
# Gitarrensolo (Takt 18–20)
_g(18, [(0.0, 2.0, A5, -2), (2.0, 0.25, C6), (2.25, 0.25, A5), (2.5, 0.25, G5),
        (2.75, 0.25, E5), (3.0, 0.5, G5), (3.5, 0.5, E5)])
_g(19, [(0.0, 0.5, D5), (0.5, 0.5, E5), (1.0, 1.0, G5, -2)])
_g(19, [(2.0, 2.0, E5)], gain=0.45)
_g(20, [(0.0, 0.25, A5), (0.25, 0.25, G5), (0.5, 0.25, E5), (0.75, 0.25, D5),
        (1.0, 0.25, C5), (1.25, 0.25, A4), (1.5, 0.5, C5), (2.0, 2.0, A5, -2)])
# Schlussakkord
_g(21, [(0.0, 5.5, E5)], gain=0.5)

# --- Abschnitte -------------------------------------------------------------
SECTIONS = [  # (Name, Start-Takt, End-Takt)
    ('intro', 0, 2), ('riff', 2, 6), ('verse', 6, 14), ('chorus', 14, 18),
    ('solo', 18, 21), ('finale', 21, 23),
]


def section_at(t):
    bar = t / BAR
    for name, b0, b1 in SECTIONS:
        if b0 <= bar < b1:
            return name
    return 'finale'


def kick_times():
    ts = []
    for bar in range(2, 21):
        ts += [bt(bar, 0), bt(bar, 2)]
    ts.append(bt(21, 0))
    return np.array(ts)


CRASH_TIMES = [bt(2), bt(6), bt(14), bt(18), bt(21)]
