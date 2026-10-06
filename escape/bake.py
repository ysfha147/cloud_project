"""Evaluate the performance per frame, solve the rig and bake keyframes."""
import numpy as np

from . import rigmath
from .choreo import Performance

FPS = 24


def bake_performance(rig, t_end, fps=FPS, frame_start=1, report=True):
    sk = rigmath.Skeleton(rig)
    perf = Performance(sk)
    n = int(round(t_end * fps)) + 1
    poses, ctrls = [], []
    for i in range(n):
        t = i / fps
        c = perf.controllers(t)
        ctrls.append(c)
    # keep the pelvis height smooth where the reach clamp kicked in
    poses = [rigmath.solve(sk, c) for c in ctrls]
    if report:
        bad = []
        for i, (c, P) in enumerate(zip(ctrls, poses)):
            for s in "lr":
                m = np.linalg.norm(P.joint("foot_" + s) - c["foot_" + s])
                h = np.linalg.norm(P.joint("hand_" + s) - c["hand_" + s])
                if m > 0.02 or h > 0.03:
                    bad.append((i + frame_start, s, round(float(m), 3), round(float(h), 3)))
        print("IK misses (frame, side, foot, hand):", bad[:40], "... total", len(bad))
    rigmath.bake(rig, sk, poses, frame_start=frame_start)
    return perf, sk, poses, ctrls
