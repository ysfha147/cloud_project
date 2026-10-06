"""Synthesize the soundtrack from the animation's contact events.

Everything is generated procedurally with numpy: wind ambience, a tension
score (drone, heartbeat percussion, hits), a distant prison siren, footsteps
on grass / dirt / concrete synced to the baked foot plants, chain-link rattles,
rail clanks, the landing thud and a few breaths.

Usage: python scripts/make_audio.py --events build/escape_events.json --out build/audio.wav
"""
import json
import math
import sys
import wave

import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# ------------------------------------------------------------ dsp helpers
def lp(x, fc):
    """Fast FFT brick-ish lowpass with a soft knee (no python loops)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= 1.0 / (1.0 + (f / fc) ** 4)
    return np.fft.irfft(X, n)


def hp(x, fc):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= (f / fc) ** 4 / (1.0 + (f / fc) ** 4)
    return np.fft.irfft(X, n)


def bp(x, f0, q=1.0):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR) + 1e-3
    X *= 1.0 / (1.0 + (q * (f / f0 - f0 / f)) ** 2)
    return np.fft.irfft(X, n)


def env_exp(n, attack, decay):
    t = np.arange(n) / SR
    a = np.minimum(1.0, t / max(attack, 1e-4))
    return a * np.exp(-t / decay)


def reverb(x, seconds=1.8, mix=0.3, bright=4000):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 6.9 / seconds)
    ir = lp(ir, bright)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = np.fft.irfft(np.fft.rfft(x, len(x) + n) * np.fft.rfft(ir, len(x) + n), len(x) + n)[: len(x)]
    return (1 - mix) * x + mix * wet


def pan(mono, p):
    """p in [-1, 1] -> stereo (equal power)."""
    a = (p + 1) * math.pi / 4
    return np.stack([mono * math.cos(a), mono * math.sin(a)], axis=1)


class Mix:
    def __init__(self, seconds):
        self.buf = np.zeros((int(SR * seconds) + SR, 2))

    def add(self, t, sig, gain=1.0, p=0.0):
        i = int(t * SR)
        if i < 0:
            sig = sig[-i:]
            i = 0
        st = pan(sig, p) if sig.ndim == 1 else sig
        j = min(len(self.buf), i + len(st))
        if j > i:
            self.buf[i:j] += gain * st[: j - i]


# ------------------------------------------------------------ instruments
def footstep(surface, intensity):
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    if surface == "dirt":
        crunch = bp(noise, 2600, 0.7) * env_exp(n, 0.002, 0.045)
        grains = np.zeros(n)
        for _ in range(14):
            k = int(rng.uniform(0, 0.07) * SR)
            g = rng.standard_normal(200) * np.exp(-np.arange(200) / 30)
            grains[k:k + 200] += g[: n - k] * rng.uniform(0.3, 1)
        body = np.sin(2 * math.pi * 75 * t) * env_exp(n, 0.002, 0.03)
        s = 0.5 * crunch + 0.25 * hp(grains, 1500) + 0.6 * body
    elif surface == "concrete":
        s = bp(noise, 1800, 1.2) * env_exp(n, 0.001, 0.03) + 0.8 * np.sin(2 * math.pi * 95 * t) * env_exp(n, 0.001, 0.035)
    else:  # grass
        swish = bp(noise, 1900, 0.5) * env_exp(n, 0.006, 0.06)
        body = np.sin(2 * math.pi * 62 * t) * env_exp(n, 0.003, 0.04)
        s = 0.45 * swish + 0.75 * body
    return s * (0.35 + 0.12 * intensity)


def metal_rattle(length=0.7, density=60, bright=1.0):
    n = int(SR * length)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for _ in range(density):
        f = rng.uniform(1800, 7500) * bright
        k = int(rng.exponential(0.12) * SR)
        if k >= n:
            continue
        m = n - k
        dec = rng.uniform(0.01, 0.05)
        s[k:] += np.sin(2 * math.pi * f * t[:m] + rng.uniform(0, 6)) * np.exp(-t[:m] / dec) * rng.uniform(0.2, 1.0)
    clank = bp(rng.standard_normal(n), 900, 0.8) * env_exp(n, 0.001, 0.08)
    return 0.22 * s + 0.4 * clank


def pipe_clank(f0=420.0, length=1.2):
    n = int(SR * length)
    t = np.arange(n) / SR
    parts = [1.0, 2.76, 5.40, 8.93, 13.34]
    s = sum(np.sin(2 * math.pi * f0 * p * t) * np.exp(-t * (2.5 + 2.2 * i)) / (1 + i) for i, p in enumerate(parts))
    hit = bp(rng.standard_normal(n), 2500, 0.6) * env_exp(n, 0.0005, 0.01)
    return 0.35 * s + 0.4 * hit


def thud(length=0.9, f=52.0):
    n = int(SR * length)
    t = np.arange(n) / SR
    fs = f * (1 + 1.5 * np.exp(-t / 0.04))
    ph = 2 * math.pi * np.cumsum(fs) / SR
    body = np.sin(ph) * env_exp(n, 0.002, 0.18)
    dirt = lp(rng.standard_normal(n), 1800) * env_exp(n, 0.002, 0.12)
    return body + 0.5 * dirt


def whoosh(length=0.8):
    n = int(SR * length)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    e = np.sin(np.pi * t / length) ** 2
    return bp(noise, 900, 0.4) * e * 0.5


def breath(length=0.45, inhale=False):
    n = int(SR * length)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    s = bp(noise, 1100 if inhale else 750, 0.9) + 0.5 * bp(noise, 2600, 2.0)
    e = np.sin(np.pi * np.minimum(1, t / length)) ** (1.5 if inhale else 0.8)
    return s * e * 0.18


def siren(seconds, period=4.2, lo=520.0, hi=1180.0, detune=1.0):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    ph = (t / period) % 1.0
    sweep = np.where(ph < 0.55, ph / 0.55, 1 - (ph - 0.55) / 0.45)
    f = (lo + (hi - lo) * (0.5 - 0.5 * np.cos(np.pi * sweep))) * detune
    phase = 2 * math.pi * np.cumsum(f) / SR
    s = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    return s


def kick(f0=55, length=0.6):
    n = int(SR * length)
    t = np.arange(n) / SR
    fs = f0 * (1 + 2.5 * np.exp(-t / 0.03))
    return np.sin(2 * math.pi * np.cumsum(fs) / SR) * env_exp(n, 0.001, 0.22)


def braam(length=3.0, f=43.65):
    n = int(SR * length)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for d in (0.995, 1.0, 1.006, 2.0, 2.01, 3.0):
        ph = 2 * math.pi * f * d * t + rng.uniform(0, 6)
        s += 2 * ((ph / (2 * math.pi)) % 1.0) - 1  # saw
    s = lp(s, 380)
    e = np.minimum(1, t / 0.08) * np.exp(-t / (length * 0.45))
    return s * e * 0.25


def drone(seconds, f=55.0):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for k, d in enumerate((1.0, 1.5, 2.0, 2.997, 4.01)):
        s += np.sin(2 * math.pi * f * d * t + 0.3 * np.sin(2 * math.pi * 0.07 * (k + 1) * t)) / (1 + k)
    trem = 0.75 + 0.25 * np.sin(2 * math.pi * 0.18 * t)
    return s * trem


# ------------------------------------------------------------------ score
def main():
    meta = json.load(open(arg("--events")))
    out = arg("--out")
    T = float(meta["t_end"]) + 0.6
    Tc, tf, tl, ts = meta["Tc"], meta["flight"], meta["land"], meta["sprint"]
    cuts = meta["cuts"]
    mix = Mix(T)
    n = len(mix.buf)
    t = np.arange(n) / SR

    # ambience: wind + distant birds
    wind = lp(rng.standard_normal(n), 500) * (0.6 + 0.4 * np.sin(2 * math.pi * 0.11 * t + 1.0))
    wind = np.stack([wind, np.roll(wind, 900)], axis=1) * 0.08
    mix.buf += wind
    for k in range(9):
        bt = rng.uniform(0.3, 7.0)
        bl = int(SR * 0.12)
        tt = np.arange(bl) / SR
        chirp = np.sin(2 * math.pi * (3200 + 2600 * tt / 0.12) * tt) * np.sin(np.pi * tt / 0.12) ** 2
        mix.add(bt, chirp, 0.02, rng.uniform(-0.8, 0.8))

    # tension score: drone swelling towards the climb, heartbeat pulses
    dr = drone(n / SR, 55.0) * 0.05 * np.clip((t - 0.5) / 3.0, 0, 1)
    dr *= 1 + 0.8 * np.clip((t - Tc + 1.0) / 3.0, 0, 1)
    mix.buf += np.stack([dr, dr], axis=1)
    beat = 0.0
    bpm = 92.0
    while beat < T - 0.5:
        g = 0.32 if beat < Tc else 0.42
        if tf - 0.1 < beat < tl:
            beat += 60.0 / bpm
            continue  # silence while he is in the air
        mix.add(beat, kick(52), g)
        mix.add(beat + 0.18, kick(48), g * 0.55)
        bpm = 92 + 40 * min(1.0, max(0.0, (beat - 2.0) / (Tc + 2.5)))
        if beat > ts:
            bpm = 140
        beat += 60.0 / bpm
    # big hits on the key moments
    mix.add(Tc + 0.5, braam(2.5), 0.55)
    mix.add(tl, braam(3.5, 41.2), 0.8)
    mix.add(tl, kick(45, 1.2), 0.9)
    # rising riser into the jump
    rl = tf - (Tc + 2.2)
    nr = int(SR * rl)
    tt = np.arange(nr) / SR
    riser = bp(rng.standard_normal(nr), 1200, 0.3) * (tt / rl) ** 2 * 0.25
    mix.add(Tc + 2.2, riser, 1.0)

    # prison siren from the moment he reaches the top
    s_start = Tc + 1.9
    sl = T - s_start
    sir = 0.6 * siren(sl) + 0.45 * siren(sl, period=4.6, detune=0.97)
    sir = lp(sir, 2500)
    fade = np.clip(np.arange(len(sir)) / (SR * 2.5), 0, 1)
    sir *= fade * 0.11
    sir = reverb(sir, 2.6, 0.55, 2500)
    mix.add(s_start, np.stack([sir, np.roll(sir, 2400)], axis=1), 1.0)

    # contact sounds
    def shot_gain(te):
        for name, a, b in cuts:
            if a <= te < b:
                return {"CamEstablish": 0.25, "CamFence": 1.0, "CamClimb": 0.9, "CamTop": 1.0, "CamDrop": 1.0, "CamSprint": 0.85}[name]
        return 0.8

    k = 0
    for ev in meta["events"]:
        kind, te, side, surf, val = ev
        g = shot_gain(te)
        p = -0.25 if side == "l" else 0.25
        if kind == "step":
            mix.add(te, footstep(surf, float(val)), 0.9 * g, p)
            k += 1
            if k % 2 == 0 and (te < Tc or te > ts):
                mix.add(te + 0.05, breath(0.32, inhale=(k // 2) % 2 == 0), 0.7 * g)
        elif kind in ("grab", "foot_mesh"):
            mix.add(te, metal_rattle(0.9 if kind == "grab" else 0.6, 70 if kind == "grab" else 40), 0.9 * g, rng.uniform(-0.3, 0.3))
        elif kind in ("rail", "rail_step"):
            mix.add(te, pipe_clank(rng.uniform(380, 460)), 0.5 * g)
            mix.add(te, metal_rattle(0.6, 35), 0.6 * g)
        elif kind == "jump":
            mix.add(te, pipe_clank(400, 1.5), 0.6 * g)
            mix.add(te - 0.05, metal_rattle(1.2, 90), 0.8 * g)
            mix.add(te + 0.05, whoosh(tl - te), 0.8)
            mix.add(te - 0.1, breath(0.25, inhale=True), 0.9)
        elif kind == "land":
            mix.add(te, thud(1.0), 1.1)
            mix.add(te, footstep("grass", 8), 1.0)
            mix.add(te + 0.12, breath(0.5), 1.0)

    # master: gentle compression + fades
    x = mix.buf
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak * 1.6
    x = np.tanh(x) / np.tanh(1.6)
    fade_in = np.clip(t / 0.4, 0, 1)
    fade_out = np.clip((T - 0.3 - t) / 1.2, 0, 1)
    x *= (fade_in * fade_out)[:, None] * 0.89
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", out, "%.1fs" % (len(pcm) / SR))


if __name__ == "__main__":
    main()
