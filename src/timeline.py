"""Gemeinsamer Zeitplan für Song und Video: "Oye Como Wau" von Lord Billy.

Eigenkomposition im Latin-Rock-Stil (Am7–D9-Vamp, Cha-Cha-Groove, singende
Leadgitarre) als Hommage an Santana. Text und Melodie sind original.
"""
import numpy as np

BPM = 120
BEAT = 60.0 / BPM          # 0,5 s
BAR = 4 * BEAT             # 2,0 s
DURATION = 45.0
SR = 44100
FPS = 30


def bt(bar, beat=0.0):
    """Zeit in Sekunden für Takt `bar` (ab 0) und Schlag `beat` (ab 0)."""
    return bar * BAR + beat * BEAT


# --- Formanten (Hz) und Maulöffnung je Vokal --------------------------------
VOWELS = {
    'a': (800, 1250, 2600), 'e': (480, 1900, 2650), 'i': (330, 2200, 3000),
    'o': (500, 880, 2550), 'u': (350, 720, 2450), 'ae': (680, 1700, 2550),
    'oe': (470, 1450, 2400), 'ue': (320, 1650, 2350), 'w': (300, 650, 2300),
}
OPENNESS = {'a': 1.0, 'ae': 0.85, 'o': 0.75, 'e': 0.6, 'oe': 0.6,
            'u': 0.5, 'ue': 0.45, 'i': 0.42}
DIPHTHONGS = {'au': ('a', 'u'), 'ai': ('a', 'e')}

A4, B4, C5, D5, E5, Fs5, G5, A5, C6 = 69, 71, 72, 74, 76, 78, 79, 81, 84
G4 = 67

# --- Gesang: (Text-Stück, Schlag, Dauer in Schlägen, MIDI, Vokal, Extras) ----
# Jede Zeile beginnt bei Takt `bar`; die Text-Stücke ergeben aneinander-
# gehängt die angezeigte Liedzeile (Karaoke).
LINES = [
    dict(bar=5, part='warmup', syl=[
        ("Au", 1.0, 0.5, D5, 'a', {}),
        ("-uuu?", 1.5, 1.75, A5, 'u', dict(scoop=-5, glide_to=G5)),
    ]),
    dict(bar=6, part='verse', syl=[
        ("O", 0.0, 0.5, E5, 'o', {}),
        ("ye, ", 0.5, 1.0, D5, 'e', {}),
        ("mein ", 1.5, 0.5, C5, 'ai', {}),
        ("Na", 2.0, 0.5, D5, 'a', {}),
        ("me ", 2.5, 0.5, C5, 'e', {}),
        ("ist ", 3.0, 0.5, A4, 'i', {}),
        ("Bil", 3.5, 1.0, C5, 'i', {}),
        ("ly", 4.5, 2.0, A4, 'i', {}),
    ]),
    dict(bar=8, part='verse', syl=[
        ("ein ", 0.0, 0.5, A4, 'ai', {}),
        ("Chow ", 0.5, 0.5, C5, 'au', {}),
        ("Chow ", 1.0, 1.0, D5, 'au', {}),
        ("mit ", 2.0, 0.5, E5, 'i', {}),
        ("Lö", 2.5, 0.5, G5, 'oe', {}),
        ("wen", 3.0, 0.5, E5, 'e', {}),
        ("mäh", 3.5, 1.0, D5, 'ae', {}),
        ("ne", 4.5, 2.0, E5, 'e', {}),
    ]),
    dict(bar=10, part='verse', syl=[
        ("Sechs ", 0.0, 0.5, E5, 'e', {}),
        ("Jah", 0.5, 0.5, E5, 'a', {}),
        ("re ", 1.0, 1.0, D5, 'e', {}),
        ("Son", 2.0, 0.5, C5, 'o', {}),
        ("ne ", 2.5, 0.5, D5, 'e', {}),
        ("im ", 3.0, 0.5, E5, 'i', {}),
        ("Fell", 3.5, 3.0, G5, 'e', dict(glide_to=E5)),
    ]),
    dict(bar=12, part='verse', syl=[
        ("die ", 0.0, 0.5, A4, 'i', {}),
        ("Zun", 0.5, 0.5, C5, 'u', {}),
        ("ge ", 1.0, 0.5, D5, 'e', {}),
        ("so ", 1.5, 0.5, E5, 'o', {}),
        ("blau ", 2.0, 1.0, G5, 'au', {}),
        ("wie ", 3.0, 0.5, E5, 'i', {}),
        ("die ", 3.5, 0.5, D5, 'i', {}),
        ("Nacht", 4.0, 3.0, A4, 'a', {}),
    ]),
    dict(bar=14, part='chorus', syl=[
        ("O", 0.0, 0.5, E5, 'o', {}),
        ("ye ", 0.5, 0.5, G5, 'e', {}),
        ("co", 1.0, 0.5, E5, 'o', {}),
        ("mo ", 1.5, 0.5, D5, 'o', {}),
        ("wau, ", 2.0, 1.5, E5, 'au', {}),
        ("wau-", 4.0, 0.5, D5, 'au', {}),
        ("wau!", 4.5, 1.75, C5, 'au', dict(glide_to=A4)),
    ]),
    dict(bar=16, part='chorus', syl=[
        ("Lord ", 0.0, 1.0, A4, 'o', {}),
        ("Bil", 1.0, 0.5, C5, 'i', {}),
        ("ly ", 1.5, 0.5, D5, 'i', {}),
        ("singt ", 2.0, 0.5, E5, 'i', {}),
        ("für ", 2.5, 0.5, G5, 'ue', {}),
        ("dich – ", 3.0, 1.0, A5, 'i', {}),
        ("au", 4.5, 0.5, E5, 'a', {}),
        ("uuu!", 5.0, 2.5, A5, 'u', dict(scoop=-7)),
    ]),
    dict(bar=19, part='solo', syl=[
        ("Au", 2.0, 0.5, E5, 'a', {}),
        ("-uuuu!", 2.5, 1.5, A5, 'u', dict(scoop=-5)),
    ]),
    dict(bar=21, part='finale', syl=[
        ("Wau", 0.0, 1.0, A4, 'au', dict(glide_to=E5)),
        ("-uuuuuh!", 1.0, 3.6, A5, 'u', dict(scoop=-5, glide_to=E5, fade=True)),
    ]),
]


def vocal_notes():
    """Liste aller gesungenen Silben mit absoluten Zeiten."""
    notes = []
    for li, line in enumerate(LINES):
        for si, (txt, beat, dur, midi, vowel, extra) in enumerate(line['syl']):
            t0 = bt(line['bar'], beat)
            notes.append(dict(line=li, idx=si, text=txt, t0=t0, t1=t0 + dur * BEAT,
                              midi=midi, vowel=vowel, **extra))
    return notes


def line_spans():
    """(Start, Ende) jeder Liedzeile für die Karaoke-Anzeige."""
    spans = []
    for li, line in enumerate(LINES):
        ns = [n for n in vocal_notes() if n['line'] == li]
        spans.append((ns[0]['t0'], ns[-1]['t1']))
    return spans


def vowel_open(v):
    if v in DIPHTHONGS:
        return OPENNESS[DIPHTHONGS[v][0]]
    return OPENNESS[v]


def mouth_open(t):
    """Maulöffnung 0..1 zum Zeitpunkt t (Array), synchron zum Gesang."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for n in vocal_notes():
        t0, t1 = n['t0'], n['t1']
        v = n['vowel']
        if v in DIPHTHONGS:
            v1, v2 = DIPHTHONGS[v]
            o1, o2 = OPENNESS[v1], OPENNESS[v2]
            mid = t0 + 0.55 * (t1 - t0)
            target = np.where(t < mid, o1, o1 + (o2 - o1) * np.clip((t - mid) / (0.3 * (t1 - t0) + 1e-6), 0, 1))
        else:
            target = np.full_like(t, OPENNESS[v])
        att = np.clip((t - t0) / 0.06, 0, 1)
        att = att * att * (3 - 2 * att)
        rel = np.clip(1 - (t - (t1 - 0.04)) / 0.12, 0, 1)
        env = att * rel
        if n.get('fade'):
            env = env * np.clip(1 - (t - t0) / ((t1 - t0) * 1.15), 0.25, 1)
        # lange Töne: leichtes Vibrato-Flattern im Maul
        flutter = 1 + 0.07 * np.sin(2 * np.pi * 5.5 * (t - t0)) * np.clip((t - t0 - 0.25) / 0.3, 0, 1)
        out = np.maximum(out, np.where((t >= t0) & (t < t1 + 0.1), env * target * flutter, 0))
    return np.clip(out, 0, 1.1)


# --- Leadgitarre: (Schlag ab Takt 2, Dauer, MIDI, Bend-Start in Halbtönen, Lautstärke)
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
