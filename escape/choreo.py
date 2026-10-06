"""The escape, as one continuous performance.

Timeline (seconds):
  0 .. Tc        run across the yard and turn to the fence (procedural run)
  Tc .. Tc+4.6   step on the plinth, jump-grab the chain-link, climb, mantle
                 the top rail, perch on it, leap down, land in a crouch
  Tc+4.4 ..      blend into a sprint away from the prison
"""
import math

import numpy as np

from .environment import FENCE_PLINTH, FENCE_TOP
from .locomotion import BodyInfo, Run, smoothstep
from .rigmath import body_rot, solve, unit

X0 = 1.4          # where he climbs, along the fence
MESH_Y = 0.045    # chain-link plane (prison side of the posts)
RAIL_Z = FENCE_TOP
G = 9.81


# ------------------------------------------------------------------ tracks
class Track:
    """Monotone cubic Hermite through (t, value[, velocity]) keys."""

    def __init__(self, keys):
        keys = sorted(keys, key=lambda k: k[0])
        self.t = np.array([k[0] for k in keys], float)
        self.v = np.array([np.atleast_1d(np.asarray(k[1], float)) for k in keys])
        n, d = self.v.shape
        m = np.zeros((n, d))
        h = np.diff(self.t)
        delta = np.diff(self.v, axis=0) / h[:, None] if n > 1 else np.zeros((0, d))
        for i in range(n):
            if i == 0 or i == n - 1:
                m[i] = 0.0
                continue
            for j in range(d):
                a, b = delta[i - 1, j], delta[i, j]
                if a * b <= 0:
                    m[i, j] = 0.0
                else:
                    w1 = 2 * h[i] + h[i - 1]
                    w2 = h[i] + 2 * h[i - 1]
                    m[i, j] = (w1 + w2) / (w1 / a + w2 / b)
        for i, k in enumerate(keys):
            if len(k) > 2 and k[2] is not None:
                m[i] = np.atleast_1d(np.asarray(k[2], float))
        self.m = m

    def __call__(self, t):
        T = self.t
        if t <= T[0]:
            return self.v[0] + self.m[0] * (t - T[0]) * 0
        if t >= T[-1]:
            return self.v[-1].copy()
        i = int(np.searchsorted(T, t) - 1)
        h = T[i + 1] - T[i]
        u = (t - T[i]) / h
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        return h00 * self.v[i] + h10 * h * self.m[i] + h01 * self.v[i + 1] + h11 * h * self.m[i + 1]


def V(*a):
    return np.array(a, float)


# ------------------------------------------------------------- performance
class Performance:
    def __init__(self, sk):
        self.sk = sk
        self.b = BodyInfo(sk)
        self.events = []
        self._plan_run()
        self._plan_climb()
        self._plan_sprint()

    # ---------------------------------------------------------------- run 1
    def _plan_run(self):
        path = [(-34.0, 11.0), (-26.0, 6.2), (-17.0, 3.2), (-8.0, 2.6), (-2.6, 2.5), (0.5, 2.05), (X0, 1.05), (X0, 0.2)]
        speed = [(0.0, 4.7), (1.5, 5.4), (5.0, 5.6), (6.2, 4.7), (7.6, 4.2)]
        best = None
        for k in range(40):
            ph0 = k * 0.05
            run = Run(self.b, path, 0.0, 9.0, speed, phase0=ph0, step_width=0.13)
            # last right-foot plant at a good distance in front of the plinth
            cand = [s for s in run.steps if s["side"] == "r" and 0.5 < s["ground"][1] < 0.85 and abs(s["ground"][0] - X0) < 0.4]
            nxt_l = [s for s in run.steps if s["side"] == "l" and cand and s["t"] > cand[-1]["t"]]
            if cand and (not nxt_l or nxt_l[0]["ground"][1] < 0.45):
                score = abs(cand[-1]["ground"][1] - 0.66)
                if best is None or score < best[0]:
                    best = (score, ph0, run, cand[-1])
        if best is None:
            raise RuntimeError("could not place the last step before the fence")
        _, ph0, run, last = best
        self.run1 = run
        self.last_plant = last
        self.Tc = last["t"] + 0.06
        for s in run.steps:
            if s["t"] <= self.Tc:
                self.events.append(("step", s["t"], s["side"], "grass", s["v"]))

    # ---------------------------------------------------------------- climb
    def _plan_climb(self):
        Tc = self.Tc
        sk = self.sk
        b = self.b
        dt = 1.0 / 48
        c0 = self.run1.controllers(Tc)
        cm = self.run1.controllers(Tc - dt)
        P0 = solve(sk, dict(c0))
        vel = lambda key: (np.asarray(c0[key]) - np.asarray(cm[key])) / dt  # noqa: E731
        T = lambda tau: Tc + tau  # noqa: E731
        ah = b.ankle_h
        pz = FENCE_PLINTH
        rz = RAIL_Z

        # pelvis -----------------------------------------------------------
        self.flight_t0 = T(3.22)
        p_take = V(X0, -0.30, 3.48)
        v_take = V(0.0, -2.4, 0.9)
        self.flight = (p_take, v_take)
        land_z = 0.98
        a_, b_, c_ = -0.5 * G, v_take[2], p_take[2] - land_z
        dl = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
        self.t_land = self.flight_t0 + dl
        p_land = p_take + v_take * dl + V(0, 0, -0.5 * G * dl * dl)
        v_land = v_take + V(0, 0, -G * dl)
        self.p_land = p_land
        tl = self.t_land - Tc

        def ballistic(tau):
            d = T(tau) - self.flight_t0
            return p_take + v_take * d + V(0, 0, -0.5 * G * d * d)

        Lx = lambda dx: X0 + dx  # noqa: E731  left limbs are on +X (he faces -Y)
        Rx_ = lambda dx: X0 - dx  # noqa: E731
        pel = [
            (T(0.0), c0["pelvis_pos"], vel("pelvis_pos") * 0.55),
            (T(0.17), V(X0, 0.38, 1.1)),
            (T(0.36), V(X0, 0.32, 1.47)),
            (T(0.50), V(X0 - 0.02, 0.34, 1.56)),
            (T(0.66), V(X0 - 0.02, 0.37, 1.36)),
            (T(0.98), V(X0, 0.35, 1.62)),
            (T(1.28), V(X0 + 0.02, 0.34, 1.92)),
            (T(1.58), V(X0 - 0.02, 0.32, 2.2)),
            (T(1.88), V(X0, 0.26, 2.52)),
            (T(2.18), V(X0, 0.16, 2.92)),
            (T(2.55), V(X0 + 0.02, 0.2, 3.12)),
            (T(2.86), V(X0, 0.02, 3.26)),
            (T(3.06), V(X0, -0.02, 3.2), V(0, -0.5, 0)),
            (T(3.22), p_take, v_take),
        ]
        for k in np.arange(3.32, tl - 0.02, 0.08):
            pel.append((T(k), ballistic(k), v_take + V(0, 0, -G * (T(k) - self.flight_t0))))
        pel += [
            (T(tl), p_land, v_land * V(1, 1, 0.55)),
            (T(tl + 0.24), V(X0, p_land[1] - 0.24, 0.56)),
            (T(tl + 0.5), V(X0 - 0.04, p_land[1] - 0.5, 0.66), V(0, -1.4, 0.5)),
            (T(tl + 0.75), V(X0 - 0.06, p_land[1] - 1.0, 0.82)),
        ]
        self.tr_pelvis = Track(pel)

        e0 = c0["_e"]

        def unwrap(h):
            while h > math.pi:
                h -= 2 * math.pi
            while h < -math.pi:
                h += 2 * math.pi
            return h

        hd0 = unwrap(e0["pelvis"][0])
        self.tr_pelvis_e = Track([
            (T(0.0), V(hd0, e0["pelvis"][1], e0["pelvis"][2])),
            (T(0.17), V(0.0, 0.2, 0.0)),
            (T(0.36), V(0.0, 0.05, 0.0)),
            (T(0.66), V(0.0, 0.1, -0.05)),
            (T(1.28), V(0.0, 0.12, 0.06)),
            (T(1.58), V(0.0, 0.15, -0.06)),
            (T(2.18), V(0.0, 0.6, 0.0)),
            (T(2.55), V(0.15, 0.75, -0.1)),
            (T(2.86), V(0.0, 0.5, 0.0)),
            (T(3.06), V(0.0, 0.7, 0.0)),
            (T(3.22), V(0.0, 0.25, 0.0)),
            (T(3.7), V(0.0, -0.05, 0.0)),
            (T(tl), V(0.0, 0.25, 0.0)),
            (T(tl + 0.24), V(0.0, 0.75, 0.0)),
            (T(tl + 0.5), V(0.0, 0.45, 0.0)),
            (T(tl + 0.75), V(0.0, 0.3, 0.0)),
        ])
        hc0 = unwrap(e0["chest"][0])
        self.tr_chest_e = Track([
            (T(0.0), V(hc0, e0["chest"][1], e0["chest"][2])),
            (T(0.17), V(0.0, 0.18, 0.0)),
            (T(0.36), V(0.0, -0.12, 0.0)),
            (T(0.66), V(0.0, -0.05, 0.0)),
            (T(0.98), V(-0.05, 0.0, -0.08)),
            (T(1.28), V(0.05, 0.05, 0.08)),
            (T(1.58), V(-0.05, 0.05, -0.08)),
            (T(1.88), V(0.0, 0.3, 0.0)),
            (T(2.18), V(0.0, 0.95, 0.0)),
            (T(2.55), V(0.12, 0.95, -0.12)),
            (T(2.86), V(0.0, 0.55, 0.0)),
            (T(3.06), V(0.0, 0.75, 0.0)),
            (T(3.22), V(0.0, 0.3, 0.0)),
            (T(3.7), V(0.0, 0.05, 0.0)),
            (T(tl), V(0.0, 0.35, 0.0)),
            (T(tl + 0.24), V(0.0, 0.85, 0.0)),
            (T(tl + 0.5), V(0.0, 0.5, 0.0)),
            (T(tl + 0.75), V(0.0, 0.35, 0.0)),
        ])
        hh0 = unwrap(e0["head"][0])
        self.tr_head_e = Track([
            (T(0.0), V(hh0, e0["head"][1], 0.0)),
            (T(0.2), V(0.0, -0.35, 0.0)),
            (T(0.66), V(0.0, -0.45, 0.0)),
            (T(1.6), V(0.0, -0.3, 0.0)),
            (T(2.18), V(0.0, 0.2, 0.0)),
            (T(2.55), V(-0.15, 0.45, 0.0)),
            (T(2.95), V(0.0, 0.5, 0.0)),
            (T(3.3), V(0.0, 0.35, 0.0)),
            (T(tl), V(0.0, 0.3, 0.0)),
            (T(tl + 0.3), V(0.0, 0.1, 0.0)),
            (T(tl + 0.75), V(0.0, 0.05, 0.0)),
        ])

        # hands --------------------------------------------------------------
        hl0, hr0 = P0.joint("hand_l"), P0.joint("hand_r")
        grab_l, grab_r = V(Lx(0.24), MESH_Y + 0.06, 2.24), V(Rx_(0.21), MESH_Y + 0.06, 2.38)
        rail_l, rail_r = V(Lx(0.24), 0.035, rz + 0.06), V(Rx_(0.24), 0.035, rz + 0.06)

        def fly_hand(tau, side, spread, fwd, up):
            p = ballistic(tau)
            sg = 1 if side == "l" else -1
            return p + V(sg * spread, -fwd, up)

        hand_l = [
            (T(0.0), hl0),
            (T(0.17), V(Lx(0.22), 0.3, 1.8)),
            (T(0.42), V(Lx(0.25), MESH_Y + 0.1, 2.2)),
            (T(0.50), grab_l), (T(1.18), grab_l),
            (T(1.3), V(Lx(0.26), 0.2, 2.75)),
            (T(1.42), rail_l), (T(2.8), rail_l),
            (T(2.95), V(Lx(0.4), -0.35, 3.55)),
            (T(3.08), V(Lx(0.35), 0.12, 3.25)),
            (T(3.24), V(Lx(0.45), -0.6, 3.85)),
        ]
        hand_r = [
            (T(0.0), hr0),
            (T(0.17), V(Rx_(0.2), 0.28, 1.92)),
            (T(0.42), V(Rx_(0.22), MESH_Y + 0.1, 2.36)),
            (T(0.50), grab_r), (T(0.86), grab_r),
            (T(0.97), V(Rx_(0.24), 0.22, 2.78)),
            (T(1.08), rail_r), (T(2.72), rail_r),
            (T(2.9), V(Rx_(0.42), -0.32, 3.5)),
            (T(3.08), V(Rx_(0.35), 0.12, 3.22)),
            (T(3.24), V(Rx_(0.45), -0.6, 3.88)),
        ]
        for k in np.arange(3.4, tl - 0.05, 0.1):
            w = smoothstep(3.3, tl, k)
            hand_l.append((T(k), fly_hand(k, "l", 0.55 + 0.1 * w, 0.25 - 0.1 * w, 0.75 - 0.35 * w)))
            hand_r.append((T(k), fly_hand(k, "r", 0.55 + 0.1 * w, 0.2 - 0.1 * w, 0.8 - 0.35 * w)))
        pl = p_land
        hand_l += [(T(tl), pl + V(0.45, -0.35, 0.25)), (T(tl + 0.24), V(Lx(0.3), pl[1] - 0.62, 0.42)), (T(tl + 0.6), V(Lx(0.25), pl[1] - 0.95, 0.85))]
        hand_r += [(T(tl), pl + V(-0.45, -0.3, 0.3)), (T(tl + 0.24), V(Rx_(0.32), pl[1] - 0.6, 0.48)), (T(tl + 0.6), V(Rx_(0.25), pl[1] - 0.75, 0.75))]
        self.tr_hand = {"l": Track(hand_l), "r": Track(hand_r)}
        self.tr_grip = {
            "l": Track([(T(0), V(0.6)), (T(0.3), V(0.2)), (T(0.46), V(0.15)), (T(0.52), V(1.0)), (T(1.2), V(1.0)), (T(1.26), V(0.2)),
                        (T(1.4), V(0.25)), (T(1.46), V(1.0)), (T(2.8), V(1.0)), (T(2.88), V(0.15)), (T(tl + 0.5), V(0.4))]),
            "r": Track([(T(0), V(0.6)), (T(0.3), V(0.2)), (T(0.46), V(0.15)), (T(0.52), V(1.0)), (T(0.88), V(1.0)), (T(0.94), V(0.2)),
                        (T(1.06), V(0.25)), (T(1.12), V(1.0)), (T(2.72), V(1.0)), (T(2.8), V(0.15)), (T(tl + 0.5), V(0.4))]),
        }
        out_l = V(0.8, 0.45, -0.6)
        out_r = V(-0.8, 0.45, -0.6)
        el0 = P0.joint("lowerarm_l") - 0.5 * (P0.joint("upperarm_l") + hl0)
        er0 = P0.joint("lowerarm_r") - 0.5 * (P0.joint("upperarm_r") + hr0)
        self.tr_elbow = {
            "l": Track([(T(0), unit(el0)), (T(0.3), out_l), (T(2.0), V(0.9, 0.4, -0.2)), (T(2.8), V(0.8, 0.5, -0.2)),
                        (T(3.0), V(0.5, 0.2, -0.8)), (T(tl), V(0.6, 0.4, -0.7)), (T(tl + 0.6), V(0.3, 0.8, -0.5))]),
            "r": Track([(T(0), unit(er0)), (T(0.3), out_r), (T(2.0), V(-0.9, 0.4, -0.2)), (T(2.72), V(-0.8, 0.5, -0.2)),
                        (T(2.92), V(-0.5, 0.2, -0.8)), (T(tl), V(-0.6, 0.4, -0.7)), (T(tl + 0.6), V(-0.3, 0.8, -0.5))]),
        }

        # feet ---------------------------------------------------------------
        plinth_l = V(Lx(0.09), 0.25, pz + ah + 0.005)
        mesh = lambda x, z: V(x, MESH_Y + 0.13, z)  # noqa: E731  ankle when the toes are in the mesh
        rail = lambda x: V(x, 0.0, rz + 0.021 + ah - 0.012)  # noqa: E731  foot standing on the rail
        cl = c0["foot_l"]
        foot_l = [
            (T(0.0), cl, vel("foot_l") * 0.5),
            (T(0.10), V(Lx(0.09), 0.42, pz + ah + 0.16)),
            (T(0.17), plinth_l), (T(0.30), plinth_l),
            (T(0.48), V(Lx(0.13), 0.3, 0.7)),
            (T(0.64), mesh(Lx(0.15), 0.82)), (T(1.06), mesh(Lx(0.15), 0.82)),
            (T(1.16), V(Lx(0.17), 0.32, 1.12)),
            (T(1.26), mesh(Lx(0.15), 1.36)), (T(1.76), mesh(Lx(0.15), 1.36)),
            (T(1.86), V(Lx(0.17), 0.36, 1.8)),
            (T(1.96), mesh(Lx(0.14), 2.12)), (T(2.4), mesh(Lx(0.14), 2.12)),
            (T(2.55), V(Lx(0.2), 0.32, 2.45)),
            (T(2.72), V(Lx(0.22), 0.25, 3.18)),
            (T(2.84), rail(Lx(0.13))), (T(3.16), rail(Lx(0.13))),
            (T(3.36), V(Lx(0.16), -0.48, 2.95)),
        ]
        foot_r = [
            (T(0.0), c0["foot_r"]), (T(0.24), c0["foot_r"]),
            (T(0.40), V(Rx_(0.1), 0.42, 0.55)),
            (T(0.64), mesh(Rx_(0.12), 1.06)), (T(1.38), mesh(Rx_(0.12), 1.06)),
            (T(1.48), V(Rx_(0.14), 0.34, 1.36)),
            (T(1.58), mesh(Rx_(0.12), 1.66)), (T(2.2), mesh(Rx_(0.12), 1.66)),
            (T(2.32), V(Rx_(0.16), 0.42, 2.55)),
            (T(2.44), V(Rx_(0.1), 0.18, 3.2)),
            (T(2.54), rail(Rx_(0.08))), (T(3.16), rail(Rx_(0.08))),
            (T(3.36), V(Rx_(0.16), -0.5, 2.95)),
        ]
        for k in np.arange(3.46, tl - 0.04, 0.1):
            w = smoothstep(3.4, tl - 0.05, k)
            p = ballistic(k)
            foot_l.append((T(k), p + V(0.13, 0.12 - 0.2 * w, -0.55 - 0.33 * w)))
            foot_r.append((T(k), p + V(-0.13, 0.05 - 0.18 * w, -0.6 - 0.28 * w)))
        land_l = V(Lx(0.15), p_land[1] - 0.22, ah)
        land_r = V(Rx_(0.15), p_land[1] - 0.12, ah)
        foot_l += [(T(tl - 0.015), land_l + V(0, 0, 0.01)), (T(tl), land_l)]
        foot_r += [(T(tl - 0.015), land_r + V(0, 0, 0.01)), (T(tl), land_r)]
        self.land_feet = {"l": land_l, "r": land_r}
        self.tr_foot = {"l": Track(foot_l), "r": Track(foot_r)}
        # foot yaw / pitch (+ toes down)
        self.tr_foot_e = {
            "l": Track([(T(0), V(0.0, 0.3)), (T(0.12), V(0.0, -0.05)), (T(0.17), V(0.0, 0.0)), (T(0.30), V(0.0, 0.25)),
                        (T(0.5), V(0.0, 0.5)), (T(0.64), V(0.0, 0.25)), (T(2.4), V(0.0, 0.25)), (T(2.72), V(0.0, 0.1)),
                        (T(2.84), V(0.0, 0.0)), (T(3.16), V(0.0, 0.0)), (T(3.3), V(0.0, 0.7)), (T(3.7), V(0.0, 0.3)),
                        (T(tl - 0.1), V(0.0, 0.15)), (T(tl), V(0.0, 0.0))]),
            "r": Track([(T(0), V(0.0, 0.0)), (T(0.24), V(0.0, 0.5)), (T(0.4), V(0.0, 0.6)), (T(0.64), V(0.0, 0.25)),
                        (T(2.2), V(0.0, 0.25)), (T(2.44), V(0.0, 0.1)), (T(2.54), V(0.0, 0.0)), (T(3.16), V(0.0, 0.0)),
                        (T(3.3), V(0.0, 0.7)), (T(3.7), V(0.0, 0.3)), (T(tl - 0.1), V(0.0, 0.15)), (T(tl), V(0.0, 0.0))]),
        }
        self.tr_knee = {
            "l": Track([(T(0), V(0.1, -1, 0.2)), (T(0.6), V(0.35, -1, 0.1)), (T(2.4), V(0.35, -1, 0.2)), (T(2.62), V(0.3, -0.6, 0.8)),
                        (T(2.9), V(0.25, -1, 0.4)), (T(3.4), V(0.2, -1, 0.1))]),
            "r": Track([(T(0), V(-0.1, -1, 0.2)), (T(0.6), V(-0.35, -1, 0.1)), (T(2.2), V(-0.35, -1, 0.2)), (T(2.4), V(-0.3, -0.6, 0.8)),
                        (T(2.9), V(-0.25, -1, 0.4)), (T(3.4), V(-0.2, -1, 0.1))]),
        }
        # contact events for the soundtrack
        ev = self.events
        ev.append(("step", T(0.17), "l", "concrete", 4.0))
        ev.append(("grab", T(0.50), "both", "mesh", 1.0))
        ev.append(("foot_mesh", T(0.64), "l", "mesh", 1.0))
        ev.append(("foot_mesh", T(0.64), "r", "mesh", 1.0))
        for tau, kind in ((1.08, "rail"), (1.26, "foot_mesh"), (1.42, "rail"), (1.58, "foot_mesh"), (1.96, "foot_mesh"),
                          (2.54, "rail_step"), (2.84, "rail_step"), (3.22, "jump")):
            ev.append((kind, T(tau), "-", "mesh", 1.0))
        ev.append(("land", self.t_land, "both", "grass", 1.0))
        self.climb_end = T(tl + 0.75)

    # --------------------------------------------------------------- sprint
    def _plan_sprint(self):
        t0 = self.t_land + 0.36
        p = self.p_land
        path = [(X0, p[1] - 0.5), (X0 + 0.3, p[1] - 8.0), (X0 + 1.5, p[1] - 30.0), (X0 + 4.0, p[1] - 80.0)]
        speed = [(t0, 1.2), (t0 + 0.5, 3.4), (t0 + 1.3, 5.6), (t0 + 2.3, 6.8), (t0 + 6.0, 7.0)]
        plants = {s: (V(self.land_feet[s][0], self.land_feet[s][1], 0.0), 0.0) for s in "lr"}

        def look(t, hd):
            # one glance back over the left shoulder
            g = smoothstep(self.t_land + 2.3, self.t_land + 2.6, t) * (1 - smoothstep(self.t_land + 3.05, self.t_land + 3.4, t))
            return hd + 1.9 * g

        self.sprint_look = look
        best = None
        for k in range(20):
            run = Run(self.b, path, t0, t0 + 14.0, speed, phase0=k * 0.05, step_width=0.12, lead="r", look=look,
                      init_plants=plants)
            first = run.steps[0] if run.steps else None
            if first and first["t"] > t0 + 0.08:
                score = abs(first["t"] - (t0 + 0.18))
                if best is None or score < best[0]:
                    best = (score, run)
        self.run2 = best[1]
        self.t_sprint = t0
        for s in self.run2.steps:
            self.events.append(("step", s["t"], s["side"], "grass" if s["ground"][1] > -1.9 or s["ground"][1] < -5.4 else "dirt", s["v"]))
        self.events.sort(key=lambda e: e[1])

    # ----------------------------------------------------------- evaluation
    def _climb_controllers(self, t):
        c = {}
        c["pelvis_pos"] = self.tr_pelvis(t)
        pe = self.tr_pelvis_e(t)
        c["pelvis_rot"] = body_rot(*pe)
        ce = self.tr_chest_e(t)
        c["spine_rot"] = body_rot(*ce)
        he = self.tr_head_e(t)
        c["head_rot"] = body_rot(*he)
        c["_e"] = dict(pelvis=tuple(pe), chest=tuple(ce), head=tuple(he))
        for s in "lr":
            c["hand_" + s] = self.tr_hand[s](t)
            c["elbow_" + s] = unit(self.tr_elbow[s](t))
            c["grip_" + s] = float(self.tr_grip[s](t)[0])
            c["foot_" + s] = self.tr_foot[s](t)
            ye, pe_ = self.tr_foot_e[s](t)
            c["foot_rot_" + s] = body_rot(ye, pe_, 0.0)
            c["toe_" + s] = -0.6 * max(0.0, min(pe_, 0.6)) * 0.0
            c["knee_" + s] = unit(self.tr_knee[s](t))
        return c

    def controllers(self, t):
        if t < self.Tc:
            return self.run1.controllers(t)
        cc = self._climb_controllers(t)
        if t < self.t_sprint:
            return cc
        rc = self.run2.controllers(t)
        w = smoothstep(self.t_sprint, self.t_sprint + 0.55, t)
        if w >= 1.0:
            return rc
        return blend(self.sk, cc, rc, w)


def _lerp(a, b, w):
    return np.asarray(a) * (1 - w) + np.asarray(b) * w


def blend(sk, a, b, w):
    """Blend two controller dicts (rotations via their Euler params)."""
    out = {}
    out["pelvis_pos"] = _lerp(a["pelvis_pos"], b["pelvis_pos"], w)
    ea, eb = a["_e"], b["_e"]
    pe = _lerp(_wrap_like(ea["pelvis"], eb["pelvis"]), eb["pelvis"], w)
    ce = _lerp(_wrap_like(ea["chest"], eb["chest"]), eb["chest"], w)
    he = _lerp(_wrap_like(ea["head"], eb["head"]), eb["head"], w)
    out["pelvis_rot"] = body_rot(*pe)
    out["spine_rot"] = body_rot(*ce)
    out["head_rot"] = body_rot(*he)
    out["_e"] = dict(pelvis=tuple(pe), chest=tuple(ce), head=tuple(he))
    P = solve(sk, dict(b))
    for s in "lr":
        hb = P.joint("hand_" + s)
        eb_ = P.joint("lowerarm_" + s) - 0.5 * (P.joint("upperarm_" + s) + hb)
        out["hand_" + s] = _lerp(a["hand_" + s], hb, w)
        out["elbow_" + s] = unit(_lerp(unit(a["elbow_" + s]), unit(eb_), w))
        out["grip_" + s] = (1 - w) * a.get("grip_" + s, 0.3) + w * b.get("grip_" + s, 0.3)
        out["foot_" + s] = _lerp(a["foot_" + s], b["foot_" + s], w)
        from .rigmath import slerp_mat

        out["foot_rot_" + s] = slerp_mat(a["foot_rot_" + s], b["foot_rot_" + s], w)
        out["toe_" + s] = (1 - w) * a.get("toe_" + s, 0) + w * b.get("toe_" + s, 0)
        out["knee_" + s] = unit(_lerp(unit(a["knee_" + s]), unit(b["knee_" + s]), w))
    return out


def _wrap_like(a, b):
    a = list(a)
    while a[0] - b[0] > math.pi:
        a[0] -= 2 * math.pi
    while a[0] - b[0] < -math.pi:
        a[0] += 2 * math.pi
    return a
