"""Procedural running with planted feet, heel-to-toe roll and arm swing."""
import math

import numpy as np

from .rigmath import Rx, Rz, body_rot, fwd_of, heading_of, left_of, unit


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class BodyInfo:
    """Rest-pose measurements needed to plan motion."""

    def __init__(self, sk):
        self.sk = sk
        H = sk.head
        self.pelvis0 = H["pelvis"].copy()
        self.hip_off = {s: H["thigh_" + s] - H["pelvis"] for s in "lr"}
        self.leg_len = {s: sk.limb["leg_" + s]["l1"] + sk.limb["leg_" + s]["l2"] for s in "lr"}
        self.foot = {}
        for s in "lr":
            g = sk.foot_geo[s]
            a0 = np.array([g["ankle"][0], g["ankle"][1], 0.0])
            self.foot[s] = dict(
                ankle=g["ankle"] - a0,
                ball=g["ball"] - a0,
                heel=g["heel"] - a0,
                toe=g["toe"] - a0,
                x=g["ankle"][0],
            )
        self.ankle_h = 0.5 * (self.foot["l"]["ankle"][2] + self.foot["r"]["ankle"][2])

    def foot_pose(self, s, ground, yaw, pitch):
        """Ankle position + world rotation for a foot touching ``ground``.

        ``ground`` is the ground point below the ankle when the foot is flat.
        pitch > 0 lifts the heel (pivot on the ball), pitch < 0 lifts the toes
        (pivot on the heel).
        """
        f = self.foot[s]
        Rh = Rz(yaw)
        R = Rh @ Rx(pitch)
        piv = f["ball"] if pitch > 0 else f["heel"]
        pivot_w = ground + Rh @ piv
        ankle = pivot_w + R @ (f["ankle"] - piv)
        return ankle, R


class Path:
    """Arc-length parameterised Catmull-Rom path through 2D points."""

    def __init__(self, pts, step=0.01):
        P = [np.array(p, float) for p in pts]
        P = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
        out = []
        for i in range(1, len(P) - 2):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            n = max(2, int(np.linalg.norm(p2 - p1) / 0.02))
            for t in np.linspace(0, 1, n, endpoint=False):
                t2, t3 = t * t, t * t * t
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
        out.append(P[-2])
        out = np.array(out)
        seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.pts = out
        self.length = self.s[-1]

    def point(self, s):
        s = np.clip(s, 0, self.length)
        x = np.interp(s, self.s, self.pts[:, 0])
        y = np.interp(s, self.s, self.pts[:, 1])
        return np.array([x, y])

    def tangent(self, s, h=0.25):
        a = self.point(s - h)
        b = self.point(s + h)
        if np.linalg.norm(b - a) < 1e-6:
            return np.array([1.0, 0.0])
        return unit(b - a)


def interp_keys(keys, t):
    ts = np.array([k[0] for k in keys])
    vs = np.array([k[1] for k in keys])
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    i = np.searchsorted(ts, t) - 1
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    u = u * u * (3 - 2 * u)
    return vs[i] + (vs[i + 1] - vs[i]) * u


class Run:
    """Running along a path.  Times in seconds.

    speed_keys: [(t, metres/second)], the path is followed from s0.
    """

    def __init__(self, body, path_pts, t0, t1, speed_keys, s0=0.0, phase0=0.0, step_width=0.12, toe_out=0.07,
                 lead="l", look=None, init_plants=None):
        self.b = body
        self.init_plants = init_plants or {}
        self.path = Path(path_pts)
        self.t0, self.t1 = t0, t1
        self.speed_keys = speed_keys
        self.step_width = step_width
        self.toe_out = toe_out
        self.look = look
        dt = 1.0 / 480
        pre = 1.2
        self.ts = np.arange(t0 - pre, t1 + 1.2, dt)
        v = np.array([interp_keys(speed_keys, max(t, t0)) for t in self.ts])
        self.v = v
        acc = np.gradient(v, dt)
        self.acc = acc
        s = np.cumsum(v) * dt
        s -= np.interp(t0, self.ts, s)
        self.s = s + s0
        cad = np.array([self.cadence(x) for x in v])
        ph = np.cumsum(cad) * dt
        ph -= np.interp(t0, self.ts, ph)
        self.phase = ph + phase0
        # footsteps: integer phase crossings
        k0 = int(math.floor(self.phase[0])) + 1
        k1 = int(math.floor(self.phase[-1]))
        self.steps = []
        sides = ("l", "r") if lead == "l" else ("r", "l")
        for k in range(k0, k1 + 1):
            tk = float(np.interp(k, self.phase, self.ts))
            vk = float(np.interp(tk, self.ts, v))
            st = self.stance_time(vk)
            side = sides[k % 2]
            smid = float(np.interp(tk + st * 0.45, self.ts, self.s))
            p = self.path.point(smid)
            tan = self.path.tangent(smid)
            hd = heading_of([tan[0], tan[1], 0])
            lat = left_of(hd)[:2]
            off = self.step_width / 2 * (1 if side == "l" else -1)
            ground = np.array([p[0] + lat[0] * off, p[1] + lat[1] * off, 0.0])
            yaw = hd + (self.toe_out if side == "l" else -self.toe_out)
            # the ground point below the ankle: the rest ankle sits off the path by its x
            self.steps.append(dict(k=k, t=tk, side=side, stance=st, ground=ground, yaw=yaw, v=vk))
        if self.init_plants:
            # starting from a standstill: no steps before the start time
            self.steps = [st for st in self.steps if st["t"] >= t0]
        self.by_side = {sd: [st for st in self.steps if st["side"] == sd] for sd in "lr"}

    # --- gait parameters
    @staticmethod
    def cadence(v):
        if v < 2.2:
            return 1.75 + 0.18 * v
        return min(3.7, 2.55 + 0.12 * v)

    @staticmethod
    def stance_time(v):
        if v < 2.2:
            return 0.62 * 2 / Run.cadence(v)
        return max(0.11, 0.37 - 0.032 * v)

    def speed(self, t):
        return float(np.interp(t, self.ts, self.v))

    def accel(self, t):
        return float(np.interp(t, self.ts, self.acc))

    def dist(self, t):
        return float(np.interp(t, self.ts, self.s))

    def ph(self, t):
        return float(np.interp(t, self.ts, self.phase))

    # --- feet
    def _contact_pitch(self, v):
        return -0.18 if v < 6.0 else 0.18  # heel strike when jogging, forefoot when sprinting

    def foot(self, side, t):
        """Return (ankle, rot, toe_bend, planted_flag)."""
        b = self.b
        steps = self.by_side[side]
        prev = None
        nxt = None
        for st in steps:
            if st["t"] <= t:
                prev = st
            elif nxt is None:
                nxt = st
        if prev is None and side in self.init_plants:
            g0, y0 = self.init_plants[side]
            v1 = steps[0]["v"]
            swing = 2.0 / self.cadence(v1) - self.stance_time(v1)
            prev = dict(t=steps[0]["t"] - swing - 0.4, stance=0.4, ground=np.array(g0, float), yaw=y0, v=v1, k=-1, side=side)
        if prev is None:
            prev = dict(steps[0])
            prev["t"] = steps[0]["t"] - 2.0 / self.cadence(steps[0]["v"])
            prev["ground"] = steps[0]["ground"] - np.append(self.path.tangent(self.dist(t)) * 1.5, 0)
        v = prev["v"]
        tc = self._contact_pitch(v) if prev.get("k", 0) != -1 else 0.0
        off_pitch = 0.55 + 0.05 * min(v, 8)
        rel = (t - prev["t"]) / prev["stance"]
        if rel <= 1.0 or nxt is None:
            rel = min(rel, 1.0)
            # stance: heel-strike -> flat -> heel-off around the ball
            if tc < 0:
                pitch = tc * (1 - smoothstep(0.0, 0.18, rel))
            else:
                pitch = tc * (1 - smoothstep(0.0, 0.3, rel)) + 0.03
            pitch += off_pitch * smoothstep(0.45, 1.0, rel) ** 1.4
            ank, R = b.foot_pose(side, prev["ground"], prev["yaw"], pitch)
            toe = -max(pitch, 0.0)
            return ank, R, toe, True
        # swing: from toe-off of prev to touchdown of nxt
        t_off = prev["t"] + prev["stance"]
        u = (t - t_off) / max(1e-3, nxt["t"] - t_off)
        a_off, R_off = b.foot_pose(side, prev["ground"], prev["yaw"], off_pitch)
        tc_n = self._contact_pitch(nxt["v"])
        a_on, R_on = b.foot_pose(side, nxt["ground"], nxt["yaw"], tc_n)
        vv = 0.5 * (prev["v"] + nxt["v"])
        if vv < 2.2:
            e = smoothstep(0.0, 1.0, u)
            lift = 0.09 * math.sin(math.pi * u)
        else:
            # the foot lingers behind while the heel kicks up, then swings through
            e = smoothstep(0.12, 0.93, u)
            H = min(0.6, 0.1 + 0.075 * vv)
            lift = H * math.sin(math.pi * min(1.0, u ** 0.75)) * (1 - 0.25 * u)
        pos = a_off + (a_on - a_off) * e
        pos = pos + np.array([0, 0, lift])
        # foot pitch: toes point down after push-off, flex up before contact
        pitch = off_pitch + (tc_n - off_pitch) * smoothstep(0.35, 0.92, u) + 0.35 * math.sin(math.pi * min(1, u * 1.6)) * (vv > 2.2)
        yaw = prev["yaw"] + (nxt["yaw"] - prev["yaw"]) * u
        R = Rz(yaw) @ Rx(pitch)
        toe = -off_pitch * (1 - smoothstep(0.0, 0.25, u))
        return pos, R, toe, False

    # --- whole body
    def controllers(self, t):
        b = self.b
        v = self.speed(t)
        a = self.accel(t)
        s = self.dist(t)
        p2 = self.path.point(s)
        tan = self.path.tangent(s)
        hd = heading_of([tan[0], tan[1], 0])
        ph = self.ph(t)
        run = v > 2.2
        # vertical bob: low at mid-stance, high in flight
        st_frac = min(0.9, self.stance_time(v) * self.cadence(v))
        bob_ph = 2 * math.pi * (ph - st_frac / 2)
        if run:
            base = b.pelvis0[2] - 0.035 - 0.006 * v - min(0.12, 0.03 * max(a, 0))
            amp = 0.024 + 0.003 * v
            z = base - amp * math.cos(bob_ph)
        else:
            z = b.pelvis0[2] - 0.012 + 0.018 * math.cos(bob_ph)
        osc = math.sin(math.pi * ph)  # +1 when the left leg swings forward
        lean = (0.1 + 0.035 * v + 0.07 * max(a, 0)) if run else 0.04
        lean = min(lean, 0.6)
        pel_yaw = 0.11 * osc if run else 0.07 * osc
        pel_rot = body_rot(hd + pel_yaw, 0.12 + 0.3 * lean, 0.045 * math.cos(bob_ph) * (1 if run else 0.5))
        chest_rot = body_rot(hd - 1.15 * pel_yaw, lean, -0.02 * osc)
        pos = np.array([p2[0], p2[1], z])
        c = dict(pelvis_pos=pos, pelvis_rot=pel_rot, spine_rot=chest_rot)
        c["_e"] = dict(pelvis=(hd + pel_yaw, 0.12 + 0.3 * lean, 0.045 * math.cos(bob_ph) * (1 if run else 0.5)),
                       chest=(hd - 1.15 * pel_yaw, lean, -0.02 * osc))
        # head looks along the path (or at a target)
        look_hd = hd
        if self.look:
            look_hd = self.look(t, hd)
        c["head_rot"] = body_rot(look_hd, 0.12 - 0.15 * lean, 0.0)
        c["_e"]["head"] = (look_hd, 0.12 - 0.15 * lean, 0.0)
        # feet + knees
        planted = []
        for sd in "lr":
            ank, R, toe, pl = self.foot(sd, t)
            c["foot_" + sd] = ank
            c["foot_rot_" + sd] = R
            c["toe_" + sd] = toe
            fw = R @ np.array([0, -1.0, 0])
            fw[2] = 0.0
            fw = unit(fw)
            c["knee_" + sd] = unit(fw + np.array([0, 0, 0.25]) + left_of(hd) * (0.1 if sd == "l" else -0.1))
            if pl:
                planted.append(sd)
        # keep planted legs within reach by lowering the pelvis if needed
        c["pelvis_pos"] = self._reach(c, planted)
        # arms swing opposite to the legs
        amp_u = (0.4 + 0.07 * v) if run else 0.28
        for sd in "lr":
            sw = osc if sd == "r" else -osc
            if run:
                flex = -0.1 + amp_u * sw
                elbow = 1.32 + 0.02 * v + 0.3 * sw
                c["armfk_" + sd] = (flex, elbow, 0.14, 0.22 + 0.1 * max(sw, 0))
                c["grip_" + sd] = 0.6
            else:
                flex = 0.02 + amp_u * sw
                c["armfk_" + sd] = (flex, 0.22 + 0.12 * max(sw, 0), 0.1, 0.05)
                c["grip_" + sd] = 0.25
            c["wrist_" + sd] = 0.0
        return c

    def _reach(self, c, planted):
        b = self.b
        pos = c["pelvis_pos"].copy()
        R = c["pelvis_rot"]
        for sd in planted:
            hip = pos + R @ b.hip_off[sd]
            ank = c["foot_" + sd]
            L = b.leg_len[sd] * 0.975
            dxy = np.linalg.norm((hip - ank)[:2])
            if dxy >= L:
                continue
            zmax = ank[2] + math.sqrt(L * L - dxy * dxy) - (R @ b.hip_off[sd])[2]
            pos[2] = min(pos[2], zmax)
        return pos
