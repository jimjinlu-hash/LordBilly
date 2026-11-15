"""Synthetisiert "Oye Como Wau" – Lord Billys Latin-Rock-Song (45 s, Stereo).

Alles wird hier aus Rohsignalen erzeugt: Congas, Cowbell, Clave, Shaker,
Drums, Bass, Hammond-Orgel mit Leslie, verzerrte Leadgitarre und Billys
Hunde-Gesang (Formant-Synthese).

Aufruf:  python3 src/make_song.py build/song.wav
"""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

import timeline as TL

SR = TL.SR
N = int((TL.DURATION + 0.5) * SR)
rng = np.random.default_rng(2026)


def hz(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def stereo(n=N):
    return np.zeros((n, 2), dtype=np.float64)


def add(buf, x, t, pan=0.0, gain=1.0):
    """Mono-Signal x bei Zeit t mit Panorama (-1..1) in Stereo-Puffer mischen."""
    i = int(round(t * SR))
    if i >= len(buf):
        return
    x = x[: len(buf) - i]
    l = np.cos((pan + 1) * np.pi / 4) * gain
    r = np.sin((pan + 1) * np.pi / 4) * gain
    buf[i:i + len(x), 0] += x * l
    buf[i:i + len(x), 1] += x * r


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def bandpass(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype='band', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def highpass(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype='high', fs=SR, output='sos'), x)


def lowpass(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype='low', fs=SR, output='sos'), x)


def noise(dur):
    return rng.standard_normal(int(dur * SR))


# ---------------------------------------------------------------- Percussion
def conga(kind, pitch):
    t = tvec(0.6)
    if kind == 'open':
        f = pitch * (1 + 0.08 * np.exp(-t / 0.03))
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.17)
        x += 0.3 * np.sin(2 * np.pi * 1.52 * np.cumsum(f) / SR) * np.exp(-t / 0.06)
        x[: int(0.006 * SR)] += 0.4 * bandpass(noise(0.006), 800, 5000)
        return x
    if kind == 'slap':
        x = bandpass(noise(0.6), 900, 4500) * np.exp(-t / 0.035) * 1.6
        x += 0.5 * np.sin(2 * np.pi * pitch * 1.3 * t) * np.exp(-t / 0.05)
        return x
    # gedämpft (heel/toe)
    return 0.35 * np.sin(2 * np.pi * pitch * 0.9 * t) * np.exp(-t / 0.035)


def cowbell():
    t = tvec(0.5)
    x = signal.square(2 * np.pi * 587 * t) * 0.6 + signal.square(2 * np.pi * 845 * t) * 0.6
    x = bandpass(x, 600, 3200, 2)
    return x * (0.7 * np.exp(-t / 0.05) + 0.3 * np.exp(-t / 0.22))


def clave():
    t = tvec(0.15)
    return (np.sin(2 * np.pi * 2450 * t) + 0.3 * np.sin(2 * np.pi * 3900 * t)) * np.exp(-t / 0.022)


def shaker(acc):
    t = tvec(0.09)
    env = np.minimum(t / 0.012, 1) * np.exp(-t / 0.03)
    return highpass(noise(0.09), 5500) * env * acc


def kick():
    t = tvec(0.45)
    f = 48 + 90 * np.exp(-t / 0.035)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    x[: int(0.004 * SR)] += 0.3 * highpass(noise(0.004), 2000)
    return np.tanh(1.6 * x)


def snare():
    t = tvec(0.35)
    x = bandpass(noise(0.35), 1200, 8000) * np.exp(-t / 0.11)
    x += 0.7 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.07)
    return x


def hihat(open_=False):
    d = 0.35 if open_ else 0.08
    t = tvec(d)
    return highpass(noise(d), 7000) * np.exp(-t / (0.12 if open_ else 0.022))


def crash():
    t = tvec(2.5)
    x = highpass(noise(2.5), 3500) * np.exp(-t / 0.9)
    for f in (3150, 4270, 5530, 6930):
        x += 0.08 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t / 0.7)
    return x


def timbale(pitch):
    t = tvec(0.5)
    x = np.sin(2 * np.pi * pitch * t) * np.exp(-t / 0.2)
    x += 0.45 * np.sin(2 * np.pi * pitch * 1.48 * t) * np.exp(-t / 0.12)
    x += 0.6 * bandpass(noise(0.5), 2000, 7000) * np.exp(-t / 0.02)
    return x


def make_percussion():
    perc = stereo()
    drums = stereo()
    for bar in range(0, 21):
        b = lambda beat: TL.bt(bar, beat)
        sec = TL.section_at(b(0))
        loud = sec in ('chorus', 'solo')
        ig = 1.7 if bar < 2 else 1.0   # Intro etwas präsenter
        # Congas: Tumbao
        for beat in (0.0, 0.5, 2.0, 2.5):
            add(perc, conga('mute', 260), b(beat), -0.45, 0.8 * ig)
        add(perc, conga('slap', 330), b(1.0), -0.45, 0.9 * ig)
        add(perc, conga('open', 330), b(3.0), -0.45, 1.0 * ig)
        add(perc, conga('open', 330), b(3.5), -0.45, 0.9 * ig)
        if bar % 2 == 1:
            add(perc, conga('open', 220), b(1.5), -0.6, 0.9)
        else:
            add(perc, conga('open', 220), b(3.75), -0.6, 0.6)
        if bar == 0:
            continue
        # Shaker 16tel
        for k in range(16):
            acc = 1.0 if k % 2 == 1 else 0.55
            add(perc, shaker(acc), b(k * 0.25), 0.35, 0.9 if loud else 0.7)
        # Clave 3-2
        hits = (0.0, 1.5, 3.0) if bar % 2 == 0 else (1.0, 2.0)
        for h in hits:
            add(perc, clave(), b(h), 0.55, 0.6)
        if bar < 2:
            continue
        # Cowbell
        for beat in range(4):
            add(perc, cowbell(), b(beat), 0.5, (1.0 if beat == 0 else 0.7) * (1.2 if loud else 0.8))
        if loud:
            for beat in (0.5, 1.5, 2.5, 3.5):
                add(perc, cowbell(), b(beat), 0.5, 0.45)
        # Drumset
        add(drums, kick(), b(0), 0, 1.0)
        add(drums, kick(), b(2), 0, 0.9)
        if loud:
            add(drums, kick(), b(2.5), 0, 0.6)
            add(drums, snare(), b(1), 0.05, 1.0)
            add(drums, snare(), b(3), 0.05, 1.0)
            for k in range(8):
                add(drums, hihat(), b(k * 0.5), 0.25, 0.8 if k % 2 == 0 else 0.5)
        else:
            add(drums, snare(), b(1), 0.05, 0.25)
            add(drums, snare(), b(3), 0.05, 0.3)
            for k in range(4):
                add(drums, hihat(), b(k + 0.5), 0.25, 0.45)
    # Timbale-Fills vor Refrain, Solo und Finale
    for bar, start in ((13, 2.0), (17, 3.0), (20, 3.0)):
        steps = np.arange(start, 4.0, 0.25)
        for k, beat in enumerate(steps):
            pitch = 720 if (k // 2) % 2 == 0 else 520
            add(drums, timbale(pitch), TL.bt(bar, beat), 0.3 - 0.6 * k / len(steps), 0.8 + 0.3 * k / len(steps))
    for tc in TL.CRASH_TIMES:
        add(drums, crash(), tc, -0.3, 1.0)
        add(drums, kick(), tc, 0, 0.6)
    # Schluss: großer Akzent
    add(drums, timbale(520), TL.bt(21), 0.2, 1.0)
    return perc, drums


# ---------------------------------------------------------------- Bass
def make_bass():
    buf = stereo()
    for bar in range(2, 21):
        root = 33 if bar % 2 == 0 else 38      # A1 / D2
        nxt = 38 if bar % 2 == 0 else 33
        pattern = [(0.0, 0.9, root, 0.9), (1.5, 0.45, root + 7, 0.8),
                   (2.0, 0.9, root + 12, 0.85), (3.5, 0.45, nxt, 1.0)]
        for beat, dur, m, g in pattern:
            d = dur * TL.BEAT
            t = tvec(d + 0.08)
            f = hz(m)
            x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
            env = np.minimum(t / 0.006, 1) * (0.6 + 0.4 * np.exp(-t / 0.15)) * np.clip((d + 0.08 - t) / 0.08, 0, 1)
            add(buf, np.tanh(1.5 * x * env), TL.bt(bar, beat), 0.0, g)
    # Schlusston
    t = tvec(3.0)
    x = np.sin(2 * np.pi * hz(33) * t) + 0.35 * np.sin(4 * np.pi * hz(33) * t)
    add(buf, np.tanh(1.5 * x * np.exp(-t / 1.2)), TL.bt(21), 0, 1.0)
    return lowpass(buf.T, 900).T


# ---------------------------------------------------------------- Orgel
DRAWBARS = [(0.5, 0.7), (1, 1.0), (1.5, 0.75), (2, 0.55), (3, 0.3), (4, 0.25), (6, 0.08)]
AM7 = [57, 60, 64, 67]
D9 = [54, 57, 60, 64]
AM9 = [45, 57, 60, 64, 67, 71]


def organ_chord(notes, dur, swell=False, perc=True):
    t = tvec(dur + 0.06)
    x = np.zeros_like(t)
    for m in notes:
        f = hz(m)
        for mult, a in DRAWBARS:
            x += a * np.sin(2 * np.pi * f * mult * t + rng.uniform(0, 6.28))
        if perc:  # Hammond-Percussion (2. Harmonische, abklingend)
            x += 0.6 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.12)
    env = np.minimum(t / 0.006, 1) * np.clip((dur + 0.06 - t) / 0.05, 0, 1)
    if swell:
        env *= np.clip(t / dur, 0, 1) ** 1.5
    x = x * env / len(notes)
    x[: int(0.003 * SR)] += 0.15 * highpass(noise(0.003), 3000)  # Key-Click
    return x


def leslie(mono, rate=6.3):
    t = np.arange(len(mono)) / SR
    out = np.zeros((len(mono), 2))
    for ch, ph in ((0, 0.0), (1, np.pi / 2)):
        d = (0.0012 + 0.0004 * np.sin(2 * np.pi * rate * t + ph)) * SR
        idx = np.arange(len(mono)) - d
        y = np.interp(idx, np.arange(len(mono)), mono)
        out[:, ch] = y * (1 + 0.22 * np.sin(2 * np.pi * rate * t + ph + 0.6))
    return out


def make_organ():
    mono = np.zeros(N)

    def put(x, t, g=1.0):
        i = int(t * SR)
        mono[i:i + len(x)] += g * x[: N - i]

    put(organ_chord(AM7, TL.BAR, swell=True, perc=False), TL.bt(1), 1.6)
    for bar in range(2, 21):
        ch = AM7 if bar % 2 == 0 else D9
        nxt = D9 if bar % 2 == 0 else AM7
        for beat, dur, c in ((0.0, 0.75, ch), (1.5, 0.5, ch), (2.5, 0.75, ch), (3.5, 0.45, nxt)):
            put(organ_chord(c, dur * TL.BEAT), TL.bt(bar, beat))
        # leiser Teppich darunter
        put(organ_chord(ch, TL.BAR, perc=False), TL.bt(bar), 0.25)
    fin = organ_chord(AM9, 3.0, perc=True)
    t = np.arange(len(fin)) / SR
    put(fin * np.exp(-t / 1.4), TL.bt(21), 1.2)
    return leslie(np.tanh(1.8 * mono) / 1.8)


# ---------------------------------------------------------------- Monophone Linien
def pitch_curve(notes, n_total, voice=False):
    """Erzeugt f0 (Hz), Amplitudenhüllkurve und Notenindex je Sample."""
    midi = np.full(n_total, np.nan)
    amp = np.zeros(n_total)
    owner = np.full(n_total, -1)
    for k, nt in enumerate(notes):
        t0, t1 = nt['t0'], nt['t1']
        rel = 0.12 if voice else 0.14
        i0, i1 = int(t0 * SR), min(int((t1 + rel) * SR), n_total)
        t = np.arange(i1 - i0) / SR
        d = t1 - t0
        m = np.full(len(t), float(nt['midi']))
        bend = nt.get('scoop', -1.2) if voice else nt.get('bend', 0)
        if bend:
            gl = 0.09 if voice else 0.15
            if voice and abs(bend) > 3:
                gl = 0.22
            x = np.clip(t / gl, 0, 1)
            m += bend * (1 - x) ** 2
        if nt.get('glide_to') is not None:
            gs = t0 + d * 0.6
            x = np.clip((t + t0 - gs) / (d * 0.4 + 0.1), 0, 1)
            m += (nt['glide_to'] - nt['midi']) * x * x * (3 - 2 * x)
        vib_delay = 0.18 if voice else 0.22
        depth = 0.32 if voice else 0.4
        vr = 5.3 if voice else 5.8
        ramp = np.clip((t - vib_delay) / 0.3, 0, 1)
        m += depth * ramp * np.sin(2 * np.pi * vr * t) * (d > 0.3)
        a = np.minimum(t / (0.035 if voice else 0.004), 1)
        if not voice:
            a *= 0.78 + 0.22 * np.exp(-t / 0.25)
            a *= 1 + 0.12 * np.clip((t - 0.6) / 0.8, 0, 1)   # Feedback-Bloom
        a *= np.clip((d + rel - t) / rel, 0, 1)
        if nt.get('fade'):
            a *= np.clip(1 - t / (d + rel), 0, 1) ** 0.7
        a *= nt.get('gain', 1.0)
        sl = slice(i0, i1)
        take = t >= 0
        if k + 1 < len(notes):   # nächste Note übernimmt (Legato)
            take &= (t + t0) < notes[k + 1]['t0']
        midi[sl] = np.where(take, m, midi[sl])
        owner[sl] = np.where(take, k, owner[sl])
        amp[sl] = np.maximum(amp[sl], a)   # Ausklang läuft weiter -> keine Klicks
    # Lücken in der Tonhöhe auffüllen (Phase bleibt stetig)
    idx = np.arange(n_total)
    good = ~np.isnan(midi)
    midi = np.interp(idx, idx[good], midi[good])
    if voice:   # natürliches Portamento zwischen Silben
        midi = signal.sosfiltfilt(signal.butter(2, 18, fs=SR, output='sos'), midi)
    return hz(midi), amp, owner


def additive(f0, weights_fn, active, chunk=SR // 2):
    """Additive Synthese: weights_fn(k, f0_chunk, sl) -> Amplitude der k-ten Harmonischen."""
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(len(f0))
    for s in range(0, len(f0), chunk):
        sl = slice(s, min(s + chunk, len(f0)))
        if not active[sl].any():
            continue
        fc = f0[sl]
        acc = np.zeros(sl.stop - sl.start)
        norm = np.zeros_like(acc)
        k = 1
        while True:
            if (k * fc.min()) > 9000 or k > 60:
                break
            w = weights_fn(k, fc, sl) * np.clip((9000 - k * fc) / 1500, 0, 1)
            acc += w * np.sin(k * phase[sl])
            norm += w * w
            k += 1
        out[sl] = acc / np.sqrt(norm + 1e-9)
    return out


def make_guitar():
    notes = sorted(TL.GUITAR, key=lambda n: n['t0'])
    f0, amp, owner = pitch_curve(notes, N, voice=False)
    active = amp > 1e-4
    sig = additive(f0, lambda k, fc, sl: (1.0 / k) * (1.25 if k in (2, 3) else 1.0), active)
    # Picking-Transienten
    for nt in notes:
        i = int(nt['t0'] * SR)
        p = highpass(noise(0.012), 1500) * np.exp(-np.arange(int(0.012 * SR)) / SR / 0.003)
        sig[i:i + len(p)] += 0.25 * p[: N - i]
    x = sig * amp
    x = np.tanh(5.0 * x) * 0.6 + 0.4 * np.tanh(2.0 * x)   # Overdrive
    x = highpass(x, 140, 2)
    x = lowpass(x, 4200, 4)
    x += 0.35 * bandpass(x, 1400, 2600, 2)  # "singende" Mitten
    return x


def make_voice():
    notes = TL.vocal_notes()
    f0, amp, owner = pitch_curve(notes, N, voice=True)
    active = amp > 1e-4
    t_all = np.arange(N) / SR
    # Formant-Spuren je Sample
    F = np.zeros((N, 3))
    F[:] = TL.VOWELS['u']
    for k, nt in enumerate(notes):
        sel = owner == k
        if not sel.any():
            continue
        t = t_all[sel] - nt['t0']
        d = nt['t1'] - nt['t0']
        v = nt['vowel']
        v1, v2 = TL.DIPHTHONGS.get(v, (v, v))
        a = np.array(TL.VOWELS['w'])
        b = np.array(TL.VOWELS[v1])
        c = np.array(TL.VOWELS[v2])
        x1 = np.clip(t / 0.07, 0, 1)[:, None]
        x1 = x1 * x1 * (3 - 2 * x1)
        x2 = np.clip((t - 0.55 * d) / (0.3 * d + 1e-3), 0, 1)[:, None]
        cur = a + (b - a) * x1
        cur = cur + (c - b) * x2 * x1
        F[sel] = cur * 1.15   # kleinerer "Hunde-Vokaltrakt"
    # Formantsprünge glätten (sonst Klicks bei Silbenwechsel)
    F = signal.sosfiltfilt(signal.butter(2, 30, fs=SR, output='sos'), F, axis=0)
    BW = np.array([90.0, 120.0, 180.0])

    def weights(k, fc, sl):
        f = k * fc
        h = 1.0 / k ** 0.9
        for j in range(3):
            Fj = F[sl, j]
            h = h * (Fj ** 2) / np.sqrt((Fj ** 2 - f ** 2) ** 2 + (f * BW[j]) ** 2)
        return h

    sig = additive(f0, weights, active)
    # Knurr-Einsatz ("rrwau") und Atem
    growl = np.ones(N)
    breath = np.zeros(N)
    for nt in notes:
        i0 = int(nt['t0'] * SR)
        t = np.arange(int(0.16 * SR)) / SR
        g = 1 - 0.4 * (1 - np.exp(-t / 0.01)) * np.exp(-t / 0.06) * (0.5 - 0.5 * np.cos(2 * np.pi * 42 * t))
        growl[i0:i0 + len(t)] *= g[: N - i0]
        b = np.minimum(t / 0.01, 1) * np.exp(-t / 0.04)
        breath[i0:i0 + len(t)] += b[: N - i0]
    shimmer = 1 + 0.04 * lowpass(rng.standard_normal(N), 25)
    x = sig * amp * growl * shimmer
    x += 0.05 * bandpass(rng.standard_normal(N), 1800, 6000) * (amp * 0.35 + breath) * (amp > 1e-3)
    x = highpass(x, 180)
    return x


# ---------------------------------------------------------------- Effekte & Mix
def reverb_ir(dur=2.2, decay=0.5, seed=1):
    r = np.random.default_rng(seed)
    t = tvec(dur)
    ir = np.zeros((len(t), 2))
    for ch in range(2):
        nz = lowpass(r.standard_normal(len(t)), 5500)
        ir[:, ch] = nz * np.exp(-t / decay)
        for d, g in ((0.011, 0.5), (0.019, 0.35), (0.027, 0.3), (0.041, 0.22)):
            ir[int((d + 0.003 * ch) * SR), ch] += g * 8
    ir[: int(0.015 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(axis=0))


IR = reverb_ir()


def reverb(st):
    out = np.zeros_like(st)
    mono = st.mean(axis=1)
    for ch in range(2):
        out[:, ch] = signal.fftconvolve(mono, IR[:, ch])[: len(mono)]
    return out


def pan_mono(x, pan):
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    return np.stack([x * l, x * r], axis=1)


def delay(st, sec, fb=0.3, mix=0.3, pingpong=True):
    d = int(sec * SR)
    out = st.copy()
    tap = st.copy()
    for k in range(1, 5):
        tap = np.roll(tap, d, axis=0)
        tap[:d] = 0
        tap *= fb if k > 1 else 1
        if pingpong:
            tap = tap[:, ::-1]
        out += mix * tap * (fb ** (k - 1))
    return out


def rms_active(st, thresh_db=-50):
    m = np.abs(st).mean(axis=1) if st.ndim == 2 else np.abs(st)
    env = lowpass(m, 10)
    ref = env.max()
    sel = env > ref * 10 ** (thresh_db / 20)
    return np.sqrt((st[sel] ** 2).mean())


def level(st, target_db):
    return st * (10 ** (target_db / 20) / (rms_active(st) + 1e-12))


def main(out_path):
    perc, drums = make_percussion()
    bass = make_bass()
    organ = make_organ()
    guitar = pan_mono(make_guitar(), -0.15)
    guitar = delay(guitar, TL.BEAT * 0.75, fb=0.35, mix=0.28)
    voice = pan_mono(make_voice(), 0.0)

    stems = {
        'perc': level(perc, -25), 'drums': level(drums, -24), 'bass': level(bass, -21),
        'organ': level(organ, -25), 'guitar': level(guitar, -19), 'voice': level(voice, -16.5),
    }
    sends = {'perc': 0.12, 'drums': 0.1, 'bass': 0.0, 'organ': 0.2, 'guitar': 0.3, 'voice': 0.32}
    mix = np.zeros((N, 2))
    wet = np.zeros((N, 2))
    for k, st in stems.items():
        mix += st
        wet += st * sends[k]
    mix += 0.5 * reverb(wet)

    # sanfte Bus-Kompression
    env = lowpass(np.abs(mix).max(axis=1), 8)
    thr = np.percentile(env, 70)
    gain = np.where(env > thr, (thr / (env + 1e-9)) ** 0.35, 1.0)
    mix *= gain[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() * 0.85)) * 0.85

    # Fade-in/-out und auf exakt 45 s kürzen
    n_out = int(TL.DURATION * SR)
    mix = mix[:n_out]
    t = np.arange(n_out) / SR
    mix *= np.clip(t / 0.02, 0, 1)[:, None]
    mix *= np.clip((TL.DURATION - t) / 0.9, 0, 1)[:, None] ** 1.5
    mix *= 10 ** (-1 / 20) / np.abs(mix).max()
    wavfile.write(out_path, SR, (mix * 32767).astype(np.int16))
    r = np.sqrt((mix ** 2).mean())
    print(f"geschrieben: {out_path}  Dauer {n_out / SR:.2f}s  RMS {20 * np.log10(r):.1f} dBFS")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'build/song.wav')
