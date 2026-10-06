"""Six shots that cut together into the escape sequence.

Each camera is keyframed per frame (position, aim, focus distance) from the
baked performance, with a little procedural handheld shake.  Timeline
markers switch the active camera at every cut.
"""
import math

import bpy
import numpy as np
from mathutils import Euler, Vector

from .choreo import X0
from .locomotion import smoothstep

FPS = 24


def _noise1(t, seed, freq):
    """Cheap smooth 1D value noise from a few sines."""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 2 * math.pi, 4)
    fr = freq * np.array([1.0, 1.73, 2.91, 4.37])
    amp = np.array([1.0, 0.55, 0.3, 0.15])
    return float(np.sum(amp * np.sin(2 * math.pi * fr * t + ph)) / amp.sum())


class Shot:
    def __init__(self, name, t0, t1, lens, fstop, place, shake=0.0, sensor=36.0):
        self.name, self.t0, self.t1 = name, t0, t1
        self.lens, self.fstop, self.place, self.shake = lens, fstop, place, shake
        self.sensor = sensor


def build_shots(perf, poses, t_end):
    """Return the shot list.  ``poses`` are per-frame solved poses (frame 1 = t 0)."""

    def joint(t, name="spine_02"):
        i = min(len(poses) - 1, max(0, int(round(t * FPS))))
        return np.array(poses[i].joint(name))

    def smooth_joint(t, name="spine_02", win=0.25):
        ts = np.linspace(t - win, t + win, 9)
        w = np.exp(-((ts - t) / (win * 0.6)) ** 2)
        pts = np.array([joint(x, name) for x in ts])
        return (pts * w[:, None]).sum(0) / w.sum()

    Tc, tf, tl, ts = perf.Tc, perf.flight_t0, perf.t_land, perf.t_sprint
    s1 = 3.4
    s2 = Tc - 0.32
    s3 = Tc + 2.12
    s4 = tf
    s5 = tl + 0.45

    def establish(t):
        # long-lens tracking shot from outside: the cell block fills the background
        u = smoothstep(0, s1, t)
        pos = np.array([-14.0, -21.0, 2.7]) + u * np.array([5.0, 0.6, -0.2])
        r = smooth_joint(t, win=0.6)
        aim = np.array([r[0] + 1.2, r[1], r[2] + 1.1])
        return pos, aim, np.linalg.norm(r - pos)

    def through_fence(t):
        r = smooth_joint(t, win=0.35)
        u = smoothstep(Tc - 1.0, Tc - 0.35, t)
        cx = r[0] + 0.15
        cx = (1 - u) * cx + u * (X0 - 1.6)
        pos = np.array([cx, -1.05, 0.62])
        aim = np.array([r[0] + 0.25 * (1 - u), r[1], 1.0 + 0.12 * u])
        return pos, aim, np.linalg.norm(joint(t) - pos)

    def climb(t):
        u = smoothstep(s2, s3, t)
        pos = np.array([X0 + 2.6, -3.35, 1.0]) + u * np.array([-0.35, 0.35, 0.3])
        r = smooth_joint(t, win=0.3)
        aim = np.array([X0 + 0.1, 0.25, 0.65 * r[2] + 0.75])
        return pos, aim, np.linalg.norm(joint(t, "spine_03") - pos)

    def over_top(t):
        u = smoothstep(s3, s4, t)
        pos = np.array([X0 - 1.25, -1.75, 3.05]) + u * np.array([0.15, 0.1, 0.12])
        r = smooth_joint(t, win=0.2)
        aim = np.array([X0 + 0.05, 0.05, 0.5 * r[2] + 1.55])
        return pos, aim, np.linalg.norm(joint(t, "spine_03") - pos)

    def drop(t):
        pos = np.array([X0 + 0.75, -4.7, 0.32])
        r = smooth_joint(t, win=0.12)
        u = smoothstep(s4, tl, t)
        aim = np.array([X0, -1.2 - 1.2 * u, 2.6 - 1.7 * u])
        aim = 0.6 * aim + 0.4 * np.array([r[0], r[1], r[2] + 0.3])
        return pos, aim, np.linalg.norm(joint(t, "spine_03") - pos)

    def sprint(t):
        # vehicle-mounted camera travelling backwards in front of him; he
        # slowly gains on it and finally runs past on the left of frame
        r = smooth_joint(t, win=0.3)
        ry = r[1]
        y_start = -8.3
        gap = np.interp(t, [s5, s5 + 0.9, s5 + 3.3, t_end - 0.6], [5.2, 3.9, 3.4, -1.2])
        y_follow = ry - gap
        w = smoothstep(s5 + 0.2, s5 + 0.9, t)
        cy = min(y_start, (1 - w) * y_start + w * y_follow)
        pos = np.array([X0 + 1.15, cy, 1.3])
        aim_r = np.array([r[0] - 0.15, r[1], r[2] + 0.45])
        far = np.array([X0 + 0.6, 12.0, 3.0])
        d = np.linalg.norm(r[:2] - pos[:2])
        k = smoothstep(2.4, 0.9, d) if r[1] > pos[1] else 1.0
        aim = (1 - k) * aim_r + k * far
        focus = (1 - k) * np.linalg.norm(joint(t, "spine_03") - pos) + k * 18.0
        return pos, aim, max(0.8, focus)

    return [
        Shot("CamEstablish", 0.0, s1, 85, 4.0, establish, shake=0.12),
        Shot("CamFence", s1, s2, 32, 2.8, through_fence, shake=0.5),
        Shot("CamClimb", s2, s3, 28, 4.0, climb, shake=0.4),
        Shot("CamTop", s3, s4, 32, 2.8, over_top, shake=0.45),
        Shot("CamDrop", s4, s5, 22, 5.0, drop, shake=0.6),
        Shot("CamSprint", s5, t_end, 50, 2.8, sprint, shake=0.5),
    ]


def build_cameras(scene, shots, frame_start=1):
    """Create one keyframed camera per shot and bind them with markers."""
    cams = []
    for k, sh in enumerate(shots):
        cd = bpy.data.cameras.new(sh.name)
        cd.lens = sh.lens
        cd.sensor_width = sh.sensor
        cd.clip_start = 0.05
        cd.clip_end = 1000
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = sh.fstop
        cd.dof.aperture_blades = 7
        cam = bpy.data.objects.new(sh.name, cd)
        scene.collection.objects.link(cam)
        f0 = int(round(sh.t0 * FPS)) + frame_start
        f1 = int(round(sh.t1 * FPS)) + frame_start
        # key one frame either side so motion blur at the cut is sane
        for f in range(f0 - 1, f1 + 2):
            t = (f - frame_start) / FPS
            pos, aim, focus = sh.place(t)
            d = Vector(aim) - Vector(pos)
            q = d.to_track_quat("-Z", "Y")
            e = q.to_euler()
            if sh.shake:
                a = math.radians(sh.shake)
                e = Euler((e.x + a * _noise1(t, 11 + k, 0.7), e.y + 0.5 * a * _noise1(t, 23 + k, 0.5),
                           e.z + a * _noise1(t, 37 + k, 0.6)))
            cam.location = Vector(pos)
            cam.rotation_euler = e
            cd.dof.focus_distance = float(focus)
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
            cd.keyframe_insert("dof.focus_distance", frame=f)
        m = scene.timeline_markers.new(sh.name, frame=f0)
        m.camera = cam
        cams.append(cam)
    scene.camera = cams[0]
    return cams
