"""Pose solver for the game-engine skeleton.

Controllers (pelvis/chest/head orientation, hand and foot targets, knee and
elbow poles, finger curl, toe bend) are solved into per-bone local rotations
with analytic two-bone IK, then baked as quaternion keyframes.
"""
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector

FWD = np.array([0.0, -1.0, 0.0])  # the rest pose faces -Y


# ------------------------------------------------------------- small helpers
def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot_axis(axis, ang):
    a = unit(axis)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def Rx(a):
    return rot_axis((1, 0, 0), a)


def Ry(a):
    return rot_axis((0, 1, 0), a)


def Rz(a):
    return rot_axis((0, 0, 1), a)


def body_rot(heading=0.0, pitch=0.0, roll=0.0, twist=0.0):
    """World rotation for a body part.

    heading: turn left (+) about Z, pitch: lean forward (+), roll: lean to the
    character's left (+), twist: extra yaw applied in the leaned frame.
    """
    return Rz(heading) @ Rx(pitch) @ Ry(roll) @ Rz(twist)


def heading_of(direction):
    d = unit([direction[0], direction[1], 0.0])
    return math.atan2(d[0], -d[1])


def fwd_of(heading):
    return np.array([math.sin(heading), -math.cos(heading), 0.0])


def left_of(heading):
    return np.array([math.cos(heading), math.sin(heading), 0.0])


def frame(y, z):
    """Right-handed frame with columns (y x z, y, z)."""
    y = unit(y)
    z = unit(np.asarray(z) - y * np.dot(z, y))
    return np.stack([np.cross(y, z), y, z], axis=1)


def slerp_mat(a, b, t):
    qa = Matrix(a.tolist()).to_quaternion()
    qb = Matrix(b.tolist()).to_quaternion()
    return np.array(qa.slerp(qb, t).to_matrix())


def two_bone(a, t, l1, l2, pole, max_reach=0.9995):
    """Return (elbow/knee position, effective end position)."""
    d = np.asarray(t, float) - a
    dist = np.linalg.norm(d)
    dh = d / max(dist, 1e-9)
    dist = min(max(dist, abs(l1 - l2) + 1e-3), (l1 + l2) * max_reach)
    end = a + dh * dist
    cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
    cos_a = max(-1.0, min(1.0, cos_a))
    sin_a = math.sqrt(1 - cos_a * cos_a)
    pp = np.asarray(pole, float) - dh * np.dot(pole, dh)
    if np.linalg.norm(pp) < 1e-6:
        pp = np.cross(dh, [0.0, 0.0, 1.0])
    pp = unit(pp)
    mid = a + l1 * (cos_a * dh + sin_a * pp)
    return mid, end


# ------------------------------------------------------------------ skeleton
class Skeleton:
    SIDES = ("l", "r")
    FINGERS = ("thumb", "index", "middle", "ring", "pinky")

    def __init__(self, rig):
        self.rig = rig
        self.bones = rig.data.bones
        self.rest = {}
        self.rest_rot = {}
        self.head = {}
        self.tail = {}
        for b in self.bones:
            m = np.array(b.matrix_local)
            self.rest[b.name] = m
            self.rest_rot[b.name] = m[:3, :3].copy()
            self.head[b.name] = np.array(b.head_local)
            self.tail[b.name] = np.array(b.tail_local)
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in self.bones}
        self.order = [b.name for b in self._ordered()]
        H = self.head
        self.len = {n: float(np.linalg.norm(self.tail[n] - self.head[n])) for n in self.order}
        self.limb = {}
        for s in self.SIDES:
            # arm: shoulder -> elbow -> wrist, leg: hip -> knee -> ankle
            self.limb["arm_" + s] = self._limb("upperarm_" + s, "lowerarm_" + s, "hand_" + s)
            self.limb["leg_" + s] = self._limb("thigh_" + s, "calf_" + s, "foot_" + s)
        # rest frames for hands: direction to the middle knuckle, palm normal
        self.hand_rest = {}
        for s in self.SIDES:
            hd = unit(H["middle_01_" + s] - H["hand_" + s])
            across = H["index_01_" + s] - H["pinky_01_" + s]
            palm = unit(np.cross(hd, across))
            inward = np.array([-1.0 if s == "l" else 1.0, 0, 0])
            if np.dot(palm, inward) < 0:
                palm = -palm
            self.hand_rest[s] = (hd, palm)
        # rest foot geometry
        self.foot_geo = {}
        for s in self.SIDES:
            ank = H["foot_" + s]
            ball = H["ball_" + s]
            toe = self.tail["ball_" + s]
            heel = np.array([ank[0], ank[1] + 0.055, 0.0])
            ballg = np.array([ball[0], ball[1], 0.0])
            self.foot_geo[s] = dict(ankle=ank, ball=ballg, heel=heel, toe=np.array([toe[0], toe[1], 0.0]))
        self.pelvis_height = H["pelvis"][2]

    def _ordered(self):
        out = []

        def walk(b):
            out.append(b)
            for c in b.children:
                walk(c)

        for b in self.bones:
            if b.parent is None:
                walk(b)
        return out

    def _limb(self, upper, lower, end):
        a, b, c = self.head[upper], self.head[lower], self.head[end]
        y1, y2 = unit(b - a), unit(c - b)
        n = np.cross(y1, y2)
        if np.linalg.norm(n) < 1e-4:
            n = np.cross(y1, FWD)
        n = unit(n)
        return dict(upper=upper, lower=lower, end=end, l1=float(np.linalg.norm(b - a)), l2=float(np.linalg.norm(c - b)),
                    n=n, F1=frame(y1, np.cross(n, y1)), F2=frame(y2, np.cross(n, y2)))


class Pose:
    """World-space bone matrices for one frame, built top-down."""

    def __init__(self, sk):
        self.sk = sk
        self.M = {}

    def place(self, name, world_delta, head=None):
        """Set bone world matrix from a world rotation delta (applied to rest)."""
        sk = self.sk
        R = world_delta @ sk.rest_rot[name]
        if head is None:
            p = sk.parent[name]
            if p is None:
                head = sk.head[name]
            else:
                local = np.linalg.solve(sk.rest[p], np.append(sk.head[name], 1.0))
                head = (self.M[p] @ local)[:3]
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = head
        self.M[name] = m
        return m

    def place_local(self, name, local_rot):
        """Set bone from a rotation in its own rest frame (fingers, toes)."""
        sk = self.sk
        p = sk.parent[name]
        rel = np.linalg.solve(sk.rest[p], sk.rest[name])
        m = self.M[p] @ rel
        m[:3, :3] = m[:3, :3] @ local_rot
        self.M[name] = m
        return m

    def world_delta(self, name):
        return self.M[name][:3, :3] @ self.sk.rest_rot[name].T

    def joint(self, name):
        return self.M[name][:3, 3].copy()

    def basis(self, name):
        sk = self.sk
        p = sk.parent[name]
        if p is None:
            ref = sk.rest[name]
        else:
            ref = self.M[p] @ np.linalg.solve(sk.rest[p], sk.rest[name])
        return np.linalg.solve(ref, self.M[name])


# ------------------------------------------------------------- solve a frame
def solve(sk, c):
    """Solve controller dict ``c`` into a Pose.

    Keys: pelvis_pos, pelvis_rot, spine_rot (chest), neck_rot, head_rot,
    clav_l/clav_r (world deltas), hand_l/hand_r (wrist positions),
    hand_rot_l/r (world delta or None to follow the forearm), elbow_l/r
    (pole directions), foot_l/r (ankle positions), foot_rot_l/r,
    knee_l/r (pole directions), toe_l/r (bend angle), grip_l/r (0..1).
    """
    P = Pose(sk)
    I = np.eye(3)
    P.place("Root", I, head=np.zeros(3))
    P.place("pelvis", c["pelvis_rot"], head=c["pelvis_pos"])
    chest = c["spine_rot"]
    pel = c["pelvis_rot"]
    P.place("spine_01", slerp_mat(pel, chest, 0.35))
    P.place("spine_02", slerp_mat(pel, chest, 0.7))
    P.place("spine_03", chest)
    P.place("neck_01", c.get("neck_rot", slerp_mat(chest, c["head_rot"], 0.45)))
    P.place("head", c["head_rot"])
    for s in sk.SIDES:
        P.place("clavicle_" + s, c.get("clav_" + s, chest))
        _arm(sk, P, s, c)
        _leg(sk, P, s, c)
    return P


def _limb_pose(sk, P, key, target, pole, twist=0.0):
    L = sk.limb[key]
    a = P.place(L["upper"], np.eye(3))[:3, 3]  # head position from the parent chain
    mid, end = two_bone(a, target, L["l1"], L["l2"], pole)
    y1, y2 = unit(mid - a), unit(end - mid)
    n = np.cross(y1, y2)
    if np.linalg.norm(n) < 1e-5:
        n = np.cross(y1, pole)
    n = unit(n)
    F1 = frame(y1, np.cross(n, y1))
    F2 = frame(y2, np.cross(n, y2))
    D1 = F1 @ L["F1"].T
    D2 = F2 @ L["F2"].T
    if twist:
        D2 = rot_axis(y2, twist) @ D2
    P.place(L["upper"], D1, head=a)
    P.place(L["lower"], D2, head=mid)
    return end


def arm_fk_target(sk, P, s, chest, flex, elbow, abduct=0.15, inward=0.2):
    """Wrist target + elbow pole from sagittal arm angles in the chest frame.

    flex: shoulder flexion (+ forward), elbow: elbow flexion, abduct: arm out
    to the side, inward: forearm rotated towards the body midline.
    """
    S = P.place("upperarm_" + s, np.eye(3))[:3, 3]
    sg = 1.0 if s == "l" else -1.0
    down = np.array([0.0, 0.0, -1.0])
    du = Ry(-sg * abduct) @ Rx(-flex) @ down
    df = Ry(-sg * abduct) @ Rz(-sg * inward) @ Rx(-(flex + elbow)) @ down
    L = sk.limb["arm_" + s]
    E = S + L["l1"] * (chest @ du)
    W = E + L["l2"] * (chest @ df)
    return W, E - (S + W) / 2


def _arm(sk, P, s, c):
    fk = c.get("armfk_" + s)
    if fk is not None:
        c["hand_" + s], c["elbow_" + s] = arm_fk_target(sk, P, s, c["spine_rot"], *fk)
    end = _limb_pose(sk, P, "arm_" + s, c["hand_" + s], c["elbow_" + s], c.get("twist_" + s, 0.0))
    hr = c.get("hand_rot_" + s)
    if hr is None:
        # follow the forearm, optionally flexing the wrist about its rest hinge
        D = P.world_delta("lowerarm_" + s)
        hr = rot_axis(D @ np.cross(*sk.hand_rest[s]), c.get("wrist_" + s, 0.0)) @ D
    P.place("hand_" + s, hr, head=end)
    grip = c.get("grip_" + s, 0.15)
    hd, palm = sk.hand_rest[s]
    for f in sk.FINGERS:
        for i in (1, 2, 3):
            name = "%s_0%d_%s" % (f, i, s)
            if name not in sk.rest:
                continue
            y = sk.rest_rot[name][:, 1]
            axis_w = unit(np.cross(y, palm))
            axis_l = sk.rest_rot[name].T @ axis_w
            if f == "thumb":
                ang = grip * (0.35, 0.55, 0.6)[i - 1]
            else:
                ang = grip * (1.25, 1.45, 1.0)[i - 1] + 0.06
            P.place_local(name, rot_axis(axis_l, ang))


def _leg(sk, P, s, c):
    end = _limb_pose(sk, P, "leg_" + s, c["foot_" + s], c["knee_" + s])
    P.place("foot_" + s, c["foot_rot_" + s], head=end)
    toe = c.get("toe_" + s, 0.0)
    # bend the toes about the lateral axis of the foot (in its rest frame)
    axis_l = sk.rest_rot["ball_" + s].T @ np.array([1.0, 0.0, 0.0])
    P.place_local("ball_" + s, rot_axis(axis_l, toe))


# ------------------------------------------------------------- baking keys
def bake(rig, sk, poses, frame_start=1):
    """Write per-frame quaternion (and pelvis location) keys in bulk."""
    action = rig.animation_data_create()
    import bpy

    act = bpy.data.actions.new(rig.name + "Action")
    rig.animation_data.action = act
    nf = len(poses)
    frames = np.arange(frame_start, frame_start + nf, dtype=np.float32)
    names = [n for n in sk.order]
    quats = {n: np.zeros((nf, 4)) for n in names}
    locs = {n: np.zeros((nf, 3)) for n in names}
    for i, P in enumerate(poses):
        for n in names:
            b = P.basis(n)
            q = Matrix(b[:3, :3].tolist()).to_quaternion()
            if i and np.dot(quats[n][i - 1], q) < 0:
                q = -q
            quats[n][i] = q
            locs[n][i] = b[:3, 3]
    for n in names:
        pb = rig.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=frame_start)
        pb.keyframe_insert("location", frame=frame_start)
    fcs = _fcurves(rig, act)
    for n in names:
        for idx in range(4):
            fc = fcs[('pose.bones["%s"].rotation_quaternion' % n, idx)]
            _fill(fc, frames, quats[n][:, idx])
        for idx in range(3):
            fc = fcs[('pose.bones["%s"].location' % n, idx)]
            _fill(fc, frames, locs[n][:, idx])
    return act


def _fcurves(rig, act):
    out = {}
    try:
        curves = act.fcurves
    except AttributeError:  # layered actions (Blender 4.4+)
        slot = rig.animation_data.action_slot
        strip = act.layers[0].strips[0]
        curves = strip.channelbag(slot).fcurves
    for fc in curves:
        out[(fc.data_path, fc.array_index)] = fc
    return out


def _fill(fc, frames, values):
    kp = fc.keyframe_points
    n = len(frames)
    if len(kp) < n:
        kp.add(n - len(kp))
    co = np.empty(2 * n, np.float32)
    co[0::2] = frames
    co[1::2] = values
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", np.full(n, 1, np.int32))  # LINEAR
    fc.update()
