"""Build the escaping prisoner from the CC0 MakeHuman base mesh (via MPFB2 data).

The base mesh, shape targets, the game-engine skeleton and its skin weights
are loaded from ``assets/mpfb``.  On top of the body we generate a khaki
prison uniform (shirt, trousers), shoes, short hair, a full beard and brows.
"""
import gzip
import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import materials as M

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "mpfb")

# Shape recipe (MakeHuman "macro" targets).  Race targets carry gender/age.
TARGETS = {
    "caucasian-male-young": 0.55,
    "african-male-young": 0.20,
    "asian-male-young": 0.25,
    "universal-male-young-maxmuscle-averageweight": 0.5,
    "universal-male-young-averagemuscle-maxweight": 0.15,
    "male-young-averagemuscle-averageweight-maxheight": 0.09,
    "male-young-averagemuscle-averageweight-idealproportions": 0.6,
}

SOLE = 0.022  # shoe sole thickness (m); the whole body is lifted by this


# --------------------------------------------------------------------------- io
def _load_obj(path):
    verts, uvs, faces, fuvs, groups = [], [], [], [], []
    group = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f "):
                vi, ti = [], []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    vi.append(int(parts[0]) - 1)
                    ti.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else 0)
                faces.append(vi)
                fuvs.append(ti)
                groups.append(group)
    return np.array(verts), np.array(uvs), faces, fuvs, groups


def _load_target(name):
    path = os.path.join(ASSETS, "targets", name + ".target.gz")
    idx, off = [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            idx.append(int(p[0]))
            off.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, int), np.array(off)


def _groups():
    raw = json.load(open(os.path.join(ASSETS, "basemesh_vertex_groups.json")))
    out = {}
    for k, ranges in raw.items():
        ids = []
        for a, b in ranges:
            ids.extend(range(a, b + 1))
        out[k] = np.array(ids, int)
    return out


def _to_blender(v):
    """MakeHuman OBJ (decimetres, Y up, facing +Z) -> Blender (metres, Z up, facing -Y)."""
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) * 0.1


# ------------------------------------------------------------------ geometry
def _vertex_normals(co, faces):
    n = np.zeros_like(co)
    for f in faces:
        p = co[f]
        # Newell normal of the polygon
        fn = np.zeros(3)
        for i in range(len(f)):
            a, b = p[i], p[(i + 1) % len(f)]
            fn += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        n[f] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def _neighbors(nv, faces):
    nb = [set() for _ in range(nv)]
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            nb[a].add(b)
            nb[b].add(a)
    return [np.array(sorted(s), int) for s in nb]


def _boundary_edges(faces):
    count = {}
    for f in faces:
        for i in range(len(f)):
            e = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            count[e] = count.get(e, 0) + 1
    return [e for e, c in count.items() if c == 1]


class Body:
    """Everything we know about the shaped base mesh in Blender space."""

    def __init__(self):
        v, uv, faces, fuvs, fgroups = _load_obj(os.path.join(ASSETS, "base.obj"))
        for name, w in TARGETS.items():
            idx, off = _load_target(name)
            v[idx] += w * off
        self.G = _groups()
        co = _to_blender(v)
        body_ids = self.G["body"]
        floor = co[body_ids, 2].min()
        co[:, 2] -= floor
        co[:, 2] += SOLE
        self.co = co
        self.uv = uv
        self.faces = faces
        self.fuvs = fuvs
        self.fgroups = fgroups
        self.height = co[body_ids, 2].max() - SOLE

    def joint(self, name):
        return self.co[self.G[name]].mean(axis=0)


# ------------------------------------------------------------------ armature
def build_armature(body, name="PrisonerRig"):
    rig = json.load(open(os.path.join(ASSETS, "rig.game_engine.json")))

    def pos(spec):
        if spec["strategy"] == "CUBE":
            return body.joint(spec["cube_name"])
        if spec["strategy"] == "MEAN":
            return body.co[spec["vertex_indices"]].mean(axis=0)
        if spec["strategy"] == "VERTEX":
            return body.co[spec["vertex_index"]]
        raise ValueError(spec["strategy"])

    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = {}
    for bname, b in rig.items():
        eb = arm.edit_bones.new(bname)
        eb.head = Vector(pos(b["head"]))
        eb.tail = Vector(pos(b["tail"]))
        if bname == "Root":
            eb.head = Vector((0, 0, 0))
            eb.tail = Vector((0, 0.25, 0))
        eb.roll = b["roll"]
        ebs[bname] = eb
    for bname, b in rig.items():
        if b["parent"]:
            ebs[bname].parent = ebs[b["parent"]]
            ebs[bname].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    arm.display_type = "STICK"
    return obj


def load_weights():
    return json.load(open(os.path.join(ASSETS, "weights.game_engine.json")))["weights"]


# ---------------------------------------------------------------- body mesh
def _make_mesh(name, co, faces, uv=None, fuvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c) for c in co], [], [tuple(f) for f in faces])
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        loop_uvs = []
        for f in fuvs:
            loop_uvs.extend(f)
        arr = uv[np.array(loop_uvs, int)].reshape(-1)
        layer.data.foreach_set("uv", arr)
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _assign_weights(obj, weights, index_map):
    """weights: bone -> [[orig_index, w]]; index_map: orig index -> new index (or -1)."""
    for bone, pairs in weights.items():
        if not pairs:
            continue
        vg = obj.vertex_groups.new(name=bone)
        by_w = {}
        for vi, w in pairs:
            ni = index_map[vi]
            if ni >= 0 and w > 1e-4:
                by_w.setdefault(round(w, 4), []).append(int(ni))
        for w, ids in by_w.items():
            vg.add(ids, w, "REPLACE")


def _attach(obj, rig, subdiv=1):
    obj.parent = rig
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    mod.use_vertex_groups = True
    if subdiv:
        s = obj.modifiers.new("Subdiv", "SUBSURF")
        s.levels = 0
        s.render_levels = subdiv
    return mod


def _store_rest(obj, co):
    attr = obj.data.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", np.asarray(co, np.float32).reshape(-1))


# ----------------------------------------------------------------- garments
def _garment(body, name, vert_mask, offset, smooth_iters, weights, shape=None, lam=0.5):
    """Create a garment shell from the body faces fully inside ``vert_mask``.

    The shell is pushed out along the normals, optionally re-shaped (``shape``
    drapes it, e.g. straight trouser legs), then Laplacian-smoothed so the
    anatomy underneath disappears while it stays outside the skin.
    """
    faces = [f for f, g in zip(body.faces, body.fgroups) if g == "body"]
    sel = [f for f in faces if all(vert_mask[i] for i in f)]
    used = np.unique(np.concatenate([np.array(f) for f in sel]))
    remap = -np.ones(len(body.co), int)
    remap[used] = np.arange(len(used))
    nfaces = [[int(remap[i]) for i in f] for f in sel]
    base = body.co[used].copy()
    nrm = _vertex_normals(base, nfaces)
    off = offset(base, used) if callable(offset) else np.full(len(used), offset)
    co = base + nrm * off[:, None]
    if shape is not None:
        co = shape(co, base, used)
    nb = _neighbors(len(co), nfaces)
    bnd, bnb, loops = _boundary(nfaces, len(co))
    for _ in range(smooth_iters):
        avg = np.array([co[n].mean(axis=0) if len(n) else co[i] for i, n in enumerate(nb)])
        bavg = np.array([co[n].mean(axis=0) if len(n) else co[i] for i, n in enumerate(bnb)])
        new = co + lam * (avg - co)
        new[bnd] = co[bnd] + lam * (bavg[bnd] - co[bnd])
        # keep the cloth outside the skin by at least 60% of the requested offset
        d = np.einsum("ij,ij->i", new - base, nrm)
        push = np.maximum(0.0, off * 0.6 - d)
        co = new + nrm * push[:, None]
    return dict(name=name, co=co, faces=nfaces, used=used, remap=remap, bnd=bnd, loops=loops, nrm=nrm)


def _boundary(faces, nv):
    bnd = np.zeros(nv, bool)
    bnb = [[] for _ in range(nv)]
    for a, b in _boundary_edges(faces):
        bnd[a] = bnd[b] = True
        bnb[a].append(b)
        bnb[b].append(a)
    loops, seen = [], set()
    for start in np.where(bnd)[0]:
        if start in seen:
            continue
        loop, prev, cur = [start], -1, start
        seen.add(start)
        while True:
            nxt = [n for n in bnb[cur] if n != prev]
            if not nxt or nxt[0] == start or nxt[0] in seen:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(np.array(loop))
    return bnd, bnb, loops


def _finish_garment(g, weights, extra_faces=None, extra_co=None, extra_src=None):
    """Turn the garment dict into a skinned object (optionally with extra rings)."""
    co, faces, remap = g["co"], list(g["faces"]), g["remap"]
    if extra_co is not None:
        base_n = len(co)
        co = np.vstack([co, extra_co])
        faces += extra_faces
    obj = _make_mesh(g["name"], co, faces)
    _store_rest(obj, co)
    _assign_weights(obj, weights, remap)
    if extra_co is not None:
        # new vertices copy the skin weights of the vertex they grew from
        me = obj.data
        for k, s_idx in enumerate(extra_src):
            groups = [(ge.group, ge.weight) for ge in me.vertices[int(s_idx)].groups]
            for gi, w in groups:
                obj.vertex_groups[gi].add([base_n + k], w, "REPLACE")
    return obj


def _inflate_segment(co, h, t, radius, weight):
    """Push points radially away from segment h-t to at least radius(s)."""
    d = t - h
    s = np.clip(((co - h) @ d) / (d @ d), 0.0, 1.0)
    c = h + s[:, None] * d
    r = co - c
    rl = np.linalg.norm(r, axis=1)
    rl[rl < 1e-6] = 1e-6
    target = np.maximum(rl, radius(s))
    newr = rl + (target - rl) * weight
    return c + r * (newr / rl)[:, None]


def _dominant_bones(weights, n):
    names = list(weights.keys())
    W = np.zeros((n, len(names)), np.float32)
    for j, bn in enumerate(names):
        for vi, w in weights[bn]:
            if vi < n:
                W[vi, j] = w
    dom = np.array(names)[W.argmax(axis=1)]
    return dom, W, names


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def build_character(collection_name="Prisoner"):
    body = Body()
    weights = load_weights()
    rig = build_armature(body)

    # ---- body + eyes
    keep_groups = {"body", "helper-l-eye", "helper-r-eye"}
    sel = [i for i, g in enumerate(body.fgroups) if g in keep_groups]
    faces = [body.faces[i] for i in sel]
    fuvs = [body.fuvs[i] for i in sel]
    fgroups = [body.fgroups[i] for i in sel]
    used = np.unique(np.concatenate([np.array(f) for f in faces]))
    remap = -np.ones(len(body.co), int)
    remap[used] = np.arange(len(used))
    nfaces = [[int(remap[i]) for i in f] for f in faces]
    body_obj = _make_mesh("PrisonerBody", body.co[used], nfaces, body.uv, fuvs)
    _store_rest(body_obj, body.co[used])
    _assign_weights(body_obj, weights, remap)
    me = body_obj.data
    skin = M.skin_material()
    eye = M.eye_material()
    me.materials.append(skin)
    me.materials.append(eye)
    mat_idx = np.array([1 if g != "body" else 0 for g in fgroups], np.int32)
    me.polygons.foreach_set("material_index", mat_idx)

    # eye direction attribute for the procedural iris
    eyedir = np.zeros((len(used), 3), np.float32)
    for side in ("l", "r"):
        ids = body.G["helper-%s-eye" % side]
        c = body.co[ids].mean(axis=0)
        for i in ids:
            if remap[i] >= 0:
                d = body.co[i] - c
                eyedir[remap[i]] = d / (np.linalg.norm(d) + 1e-9)
    a = me.attributes.new("eyedir", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", eyedir.reshape(-1))

    _attach(body_obj, rig, subdiv=1)

    # ---- landmarks
    nb = len(body.G["body"])
    dom, W, names = _dominant_bones(weights, len(body.co))
    co = body.co
    J = body.joint
    z_ankle = (J("joint-l-ankle")[2] + J("joint-r-ankle")[2]) / 2
    z_waist = J("joint-spine-4")[2] + 0.035
    z_hem = J("joint-pelvis")[2] - 0.07
    eye_l = co[body.G["helper-l-eye"]].mean(axis=0)
    eye_r = co[body.G["helper-r-eye"]].mean(axis=0)
    z_eye = (eye_l[2] + eye_r[2]) / 2
    is_body = np.zeros(len(co), bool)
    is_body[:nb] = True

    def bone_param(bone, p):
        b = rig.data.bones[bone]
        h, t = np.array(b.head_local), np.array(b.tail_local)
        d = t - h
        return ((p - h) @ d) / (d @ d)

    arm_bones = {"upperarm_l", "upperarm_r"}
    torso_bones = {"spine_01", "spine_02", "spine_03", "clavicle_l", "clavicle_r", "pelvis"}
    leg_bones = {"pelvis", "thigh_l", "thigh_r", "calf_l", "calf_r", "spine_01"}

    # ---- shirt (short sleeves, untucked, collar)
    shirt_mask = np.zeros(len(co), bool)
    for i in range(nb):
        d = dom[i]
        if d in torso_bones and co[i, 2] > z_hem:
            shirt_mask[i] = True
        elif d in arm_bones and bone_param(d, co[i]) < 0.5:
            shirt_mask[i] = True
        elif d == "neck_01" and co[i, 2] < J("joint-neck")[2] - 0.005 and co[i, 1] > J("joint-neck")[1] - 0.02:
            shirt_mask[i] = True  # back of the neck base, collar sits here

    z_chest = J("joint-spine-1")[2] - 0.05
    torso_ids = np.array([i for i in range(nb) if dom[i] in torso_bones])
    yc = float(np.median(co[torso_ids, 1]))

    def shirt_shape(cc, base, ids):
        out = cc.copy()
        dsel = dom[ids]
        # drape the torso straight down from the chest
        tor = np.isin(dsel, list(torso_bones))
        x, y, z = cc[:, 0], cc[:, 1] - yc, cc[:, 2]
        th = np.arctan2(x, -y)
        r = np.hypot(x, y)
        bins = np.linspace(-np.pi, np.pi, 49)
        bi = np.clip(np.digitize(th, bins) - 1, 0, 47)
        chest = tor & (np.abs(z - z_chest) < 0.05)
        rmax = np.zeros(48)
        for k in range(48):
            m = chest & (bi == k)
            rmax[k] = r[m].max() if m.any() else 0.0
        rmax = np.maximum(rmax, np.roll(rmax, 1) * 0.98)
        rmax = np.maximum(rmax, np.roll(rmax, -1) * 0.98)
        below = tor & (z < z_chest)
        side = np.abs(np.sin(th))
        target = rmax[bi] * (0.985 - 0.04 * side) + 0.004
        w = _smoothstep(z_chest, z_chest - 0.12, z) * below
        newr = r + np.maximum(0.0, target - r) * w
        k = np.where(below)[0]
        out[k, 0] = np.sin(th[k]) * newr[k]
        out[k, 1] = yc - np.cos(th[k]) * newr[k]
        # loose short sleeves
        for sd in ("l", "r"):
            arm = dsel == "upperarm_" + sd
            if arm.any():
                b = rig.data.bones["upperarm_" + sd]
                h, t = np.array(b.head_local), np.array(b.tail_local)
                sub = _inflate_segment(out[arm], h, t, lambda s_: 0.056 + 0.016 * s_,
                                       _smoothstep(0.08, 0.3, ((out[arm] - h) @ (t - h)) / ((t - h) @ (t - h))))
                out[arm] = sub
        return out

    def shirt_offset(base, ids):
        return np.full(len(base), 0.012)

    g = _garment(body, "PrisonerShirt", shirt_mask, shirt_offset, 25, weights, shape=shirt_shape)
    # hems flare a little, the neck loop grows a turned-down collar
    gco = g["co"]
    gn = _vertex_normals(gco, g["faces"])
    gco[g["bnd"]] += gn[g["bnd"]] * 0.005
    neck = max(g["loops"], key=lambda lp: gco[lp, 2].mean())
    ring_co, ring_faces, ring_src = _collar(gco, neck, J("joint-neck"))
    shirt = _finish_garment(g, weights, ring_faces, ring_co, np.concatenate([neck, neck]))
    shirt.data.materials.append(M.uniform_material("Shirt", shirt=True))
    _cloth_modifiers(shirt, rig, thickness=0.004, displace=0.004)

    # ---- trousers: straight, loose legs
    pants_mask = np.zeros(len(co), bool)
    for i in range(nb):
        if dom[i] in leg_bones and z_ankle + 0.035 < co[i, 2] < z_waist:
            pants_mask[i] = True
    z_crotch = co[:nb][np.abs(co[:nb, 0]) < 0.01][:, 2]
    z_crotch = z_crotch[z_crotch < J("joint-pelvis")[2]].max()

    def pants_shape(cc, base, ids):
        out = cc.copy()
        for sd in ("l", "r"):
            side = (cc[:, 0] > 0) if sd == "l" else (cc[:, 0] < 0)
            th_b = rig.data.bones["thigh_" + sd]
            ca_b = rig.data.bones["calf_" + sd]
            hip, knee, ank = np.array(th_b.head_local), np.array(ca_b.head_local), np.array(ca_b.tail_local)
            upper = side & (cc[:, 2] >= knee[2])
            lower = side & (cc[:, 2] < knee[2])
            w_up = _smoothstep(z_crotch - 0.03, z_crotch - 0.16, cc[upper, 2])
            out[upper] = _inflate_segment(cc[upper], hip, knee, lambda s_: 0.080 - 0.006 * s_, w_up)
            out[lower] = _inflate_segment(cc[lower], knee, ank, lambda s_: 0.074 + 0.004 * s_, np.ones(lower.sum()))
        return out

    def pants_offset(base, ids):
        return np.full(len(base), 0.011)

    g = _garment(body, "PrisonerPants", pants_mask, pants_offset, 22, weights, shape=pants_shape)
    gco = g["co"]
    gn = _vertex_normals(gco, g["faces"])
    gco[g["bnd"]] += gn[g["bnd"]] * 0.004
    pants = _finish_garment(g, weights)
    pants.data.materials.append(M.uniform_material("Pants", shirt=False))
    _cloth_modifiers(pants, rig, thickness=0.004, displace=0.003)

    # ---- shoes: parametric sneakers skinned to foot / toes / calf
    shoes = _sneakers(rig, weights, body, dom)

    # ---- hair / beard / brows as particle hair on the body
    hc = co[:nb]
    head_ids = np.where(np.isin(dom[:nb], ["head"]))[0]
    head_c = np.array([0.0, (eye_l[1] + eye_r[1]) / 2 + 0.085, z_eye])
    rel = hc - head_c
    ang = np.arctan2(rel[:, 0], -rel[:, 1])  # 0 = front, +-pi = back
    front = np.cos(ang)
    hairline = np.where(front > 0, z_eye + 0.020 + 0.040 * np.clip(front, 0, None) ** 1.5, z_eye + 0.020 + 0.075 * front)
    is_head = np.isin(dom[:nb], ["head"])
    ear_x = np.abs(hc[head_ids, 0]).max()
    scalp = is_head & (hc[:, 2] > hairline) & (np.abs(hc[:, 0]) < ear_x - 0.012)
    scalp_w = scalp * _smoothstep(0.0, 0.012, hc[:, 2] - hairline)

    z_lips = z_eye - 0.075
    nose_bottom = z_eye - 0.056
    y_front = hc[head_ids, 1].min()
    front_mid = hc[head_ids][(np.abs(hc[head_ids, 0]) < 0.02) & (hc[head_ids, 1] < y_front + 0.06)]
    chin_z = front_mid[:, 2].min()
    is_neck = np.isin(dom[:nb], ["neck_01"])
    ax = np.abs(hc[:, 0])
    # cheek line: from the mouth corner up to the front of the ear (sideburns)
    cheek_line = z_lips + 0.012 + np.clip((ax - 0.025) / 0.045, 0, 1) * (z_eye - 0.028 - z_lips - 0.012)
    beard = (is_head | is_neck) & (hc[:, 1] < head_c[1] + 0.01) & (hc[:, 2] < cheek_line)
    mustache = is_head & (ax < 0.032) & (hc[:, 2] > z_lips + 0.005) & (hc[:, 2] < nose_bottom) & (hc[:, 1] < y_front + 0.04)
    beard |= mustache
    beard &= hc[:, 2] > chin_z - 0.03
    lips = (ax < 0.026) & (np.abs(hc[:, 2] - z_lips) < 0.0085) & (hc[:, 1] < y_front + 0.035)
    beard &= ~lips
    beard_w = beard * (0.3 + 0.7 * _smoothstep(chin_z - 0.03, chin_z - 0.005, hc[:, 2]))
    brows = is_head & (np.abs(hc[:, 2] - (z_eye + 0.022)) < 0.0055) & (ax > 0.012) & (ax < 0.056) & (hc[:, 1] < y_front + 0.04)

    def vgroup(name, w):
        vg = body_obj.vertex_groups.new(name=name)
        for i in np.where(w > 0.01)[0]:
            ni = remap[i]
            if ni >= 0:
                vg.add([int(ni)], float(w[i]), "REPLACE")
        return vg

    vgroup("hair_scalp", scalp_w.astype(float))
    vgroup("hair_beard", beard_w.astype(float))
    vgroup("hair_brows", brows.astype(float))
    hair_mat = M.hair_material()
    body_obj.data.materials.append(hair_mat)
    hair_slot = len(body_obj.data.materials)
    _hair(body_obj, "Scalp", "hair_scalp", count=3200, length=0.013, children=24, slot=hair_slot, down=0.35, back=0.5, radius=0.00011)
    _hair(body_obj, "Beard", "hair_beard", count=3600, length=0.011, children=20, slot=hair_slot, down=0.7, back=0.0, radius=0.0001)
    _hair(body_obj, "Brows", "hair_brows", count=260, length=0.0065, children=8, slot=hair_slot, down=0.6, back=0.0, radius=0.00009)
    # particle systems must follow the armature but precede subdivision
    _reorder_modifiers(body_obj)

    coll = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(coll)
    for o in (rig, body_obj, shirt, pants, shoes):
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    info = {
        "rig": rig,
        "body": body_obj,
        "garments": [shirt, pants, shoes],
        "height": body.height,
        "z_eye": z_eye,
    }
    return info


def _reorder_modifiers(obj):
    names = [m.name for m in obj.modifiers]
    order = [n for n in names if obj.modifiers[n].type == "ARMATURE"]
    order += [n for n in names if obj.modifiers[n].type == "PARTICLE_SYSTEM"]
    order += [n for n in names if n not in order]
    for i, n in enumerate(order):
        cur = [m.name for m in obj.modifiers].index(n)
        while cur > i:
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.modifier_move_up(modifier=n)
            cur -= 1


def _hair(obj, name, vgroup, count, length, children, slot, down=0.0, back=0.15, radius=0.0001):
    """Particle hair.  With 'advanced' hair Blender grows strands to 4x the
    emission velocity, so the velocity vector encodes length and direction."""
    mod = obj.modifiers.new(name, "PARTICLE_SYSTEM")
    ps = mod.particle_system
    st = ps.settings
    st.name = name
    st.type = "HAIR"
    st.count = count
    st.hair_length = length
    st.emit_from = "FACE"
    st.use_advanced_hair = True
    v = length / 4.0
    norm = math.sqrt(max(0.0, 1.0 - down * down - back * back))
    st.normal_factor = v * norm
    st.object_align_factor = (0.0, v * back, -v * down)
    st.child_type = "INTERPOLATED"
    st.child_percent = max(1, children // 4)
    st.rendered_child_count = children
    st.child_length = 1.0
    st.roughness_1 = 0.0015
    st.roughness_1_size = 0.4
    st.roughness_endpoint = 0.001
    st.child_radius = 0.004
    st.clump_factor = 0.1
    st.material = slot
    st.root_radius = 1.0
    st.tip_radius = 0.25
    st.radius_scale = radius
    st.use_close_tip = True
    st.display_step = 2
    st.render_step = 3
    ps.vertex_group_density = vgroup
    st.use_modifier_stack = False
    return ps


def _collar(co, loop, neck_joint):
    """Two rings above the neckline: a stand band and a turned-down flap."""
    p = co[loop]
    centre = np.array([neck_joint[0], p[:, 1].mean(), 0.0])
    out = p - centre
    out[:, 2] = 0
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    up = np.array([0, 0, 1.0])
    ring1 = p + up * 0.024 + out * 0.003
    ring2 = ring1 + out * 0.026 - up * 0.027
    n = len(loop)
    base1, base2 = len(co), len(co) + n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([int(loop[i]), int(loop[j]), base1 + j, base1 + i])
        faces.append([base1 + i, base1 + j, base2 + j, base2 + i])
    return np.vstack([ring1, ring2]), faces, None


def _sneakers(rig, weights, body, dom):
    """Sneakers lofted around each foot: superellipse rings sized to the foot."""
    bones = rig.data.bones
    nbody = len(body.G["body"])
    all_co, all_faces, wts = [], [], []
    nu, nv = 28, 28
    for sd in ("l", "r"):
        ank = np.array(bones["foot_" + sd].head_local)
        ball = np.array(bones["ball_" + sd].head_local)
        fwd = ball - ank
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        lat = np.cross(fwd, [0, 0, 1.0])
        ids = [i for i in range(nbody) if dom[i] in ("foot_" + sd, "ball_" + sd) and body.co[i, 2] < ank[2] + 0.03]
        pts = body.co[ids]
        rel = pts - ank
        uu = rel @ fwd
        ll = rel @ lat
        u0, u1 = uu.min() - 0.012, uu.max() + 0.014
        length = u1 - u0
        bins = np.linspace(0, 1, nu)
        lo = np.zeros(nu)
        hi = np.zeros(nu)
        top = np.zeros(nu)
        un = (uu - u0) / length
        for i, b_ in enumerate(bins):
            m = np.abs(un - b_) < 0.06
            lo[i], hi[i], top[i] = ll[m].min(), ll[m].max(), pts[m, 2].max()
        k = np.ones(3) / 3
        lo = np.convolve(np.pad(lo, 1, mode="edge"), k, "valid")
        hi = np.convolve(np.pad(hi, 1, mode="edge"), k, "valid")
        top = np.convolve(np.pad(top, 1, mode="edge"), k, "valid")
        u_ball = ((ball - ank) @ fwd - u0) / length
        start = len(all_co)
        for i in range(nu):
            u = bins[i]
            m_lat = 0.010
            half = (hi[i] - lo[i]) / 2 + m_lat
            cen = (hi[i] + lo[i]) / 2
            tz = top[i] + 0.012
            taper = 1.0
            if i == 0 or i == nu - 1:
                taper = 0.55
            elif i == 1 or i == nu - 2:
                taper = 0.88
            centre = ank + fwd * (u0 + u * length) + lat * cen
            for j in range(nv):
                a_ = 2 * math.pi * j / nv
                ca, sa = math.cos(a_), math.sin(a_)
                ex = 2.8
                px = np.sign(ca) * abs(ca) ** (2 / ex) * half * taper
                pz = np.sign(sa) * abs(sa) ** (2 / ex)
                h = tz / 2 * (taper if i in (0, nu - 1) else 1.0)
                z = max(0.0015, tz / 2 + pz * h)
                p = centre + lat * px
                all_co.append(np.array([p[0], p[1], z]))
                if u >= u_ball + 0.03:
                    wts.append({"ball_" + sd: 1.0})
                elif u >= u_ball - 0.07:
                    t = (u - (u_ball - 0.07)) / 0.1
                    wts.append({"ball_" + sd: t, "foot_" + sd: 1 - t})
                elif z > ank[2] + 0.01 and u < 0.4:
                    wts.append({"foot_" + sd: 0.7, "calf_" + sd: 0.3})
                else:
                    wts.append({"foot_" + sd: 1.0})
        for i in range(nu - 1):
            for j in range(nv):
                a0 = start + i * nv + j
                a1 = start + i * nv + (j + 1) % nv
                all_faces.append([a0, a1, a1 + nv, a0 + nv])
        all_faces.append([start + j for j in reversed(range(nv))])
        last = start + (nu - 1) * nv
        all_faces.append([last + j for j in range(nv)])
    co = np.array(all_co)
    obj = _make_mesh("PrisonerShoes", co, all_faces)
    _store_rest(obj, co)
    groups = {}
    for i, w in enumerate(wts):
        for b_, val in w.items():
            groups.setdefault(b_, []).append((i, val))
    for b_, lst in groups.items():
        vg = obj.vertex_groups.new(name=b_)
        for i, val in lst:
            vg.add([i], val, "REPLACE")
    obj.data.materials.append(M.shoe_material())
    _attach(obj, rig, subdiv=0)
    sub = obj.modifiers.new("Subdiv", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 1
    return obj


def _cloth_modifiers(obj, rig, thickness, displace):
    if displace > 0:
        tex = bpy.data.textures.new(obj.name + "_wrinkles", "CLOUDS")
        tex.noise_scale = 0.06
        tex.noise_depth = 2
        d = obj.modifiers.new("Wrinkles", "DISPLACE")
        d.texture = tex
        d.texture_coords = "LOCAL"
        d.strength = displace
        d.mid_level = 0.5
    _attach(obj, rig, subdiv=0)
    sol = obj.modifiers.new("Thickness", "SOLIDIFY")
    sol.thickness = thickness
    sol.offset = -1.0
    sol.use_even_offset = False  # even offset spikes on the collar seam
    sub = obj.modifiers.new("Subdiv", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 1
