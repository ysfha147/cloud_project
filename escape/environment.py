"""The prison: cell block, yard, perimeter fence, trees, guard tower and sky.

World layout (metres): the perimeter fence runs along X at y = 0 on a low
concrete plinth.  The prison yard is y > 0 with the cell block facade at
y = 44; outside the fence (y < 0) there is a patrol road and a weedy field.
All shots look roughly towards +Y.
"""
import math
import random

import bpy
import numpy as np

from . import materials as M

FENCE_PLINTH = 0.45
FENCE_TOP = 2.85
FENCE_X = (-70.0, 70.0)
POST_SPACING = 3.0
BUILDING_Y = 44.0


# ------------------------------------------------------------- mesh builder
class MeshBuilder:
    def __init__(self):
        self.v, self.f, self.m, self.uv = [], [], [], []

    def _add(self, verts, faces, mat, uvs=None):
        base = len(self.v)
        self.v.extend([tuple(map(float, p)) for p in verts])
        for k, f in enumerate(faces):
            self.f.append([base + i for i in f])
            self.m.append(mat)
            self.uv.append(uvs[k] if uvs is not None else None)

    def box(self, center, size, mat=0, rot=0.0):
        cx, cy, cz = center
        sx, sy, sz = (s / 2 for s in size)
        c, s = math.cos(rot), math.sin(rot)
        pts = []
        for dz in (-sz, sz):
            for dy in (-sy, sy):
                for dx in (-sx, sx):
                    pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c, cz + dz))
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        self._add(pts, faces, mat)

    def cone(self, p0, p1, r0, r1, mat=0, segs=10, caps=True):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        ax = p1 - p0
        ax /= np.linalg.norm(ax)
        ref = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(ax, ref)
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        pts = []
        for p, r in ((p0, r0), (p1, r1)):
            for i in range(segs):
                a = 2 * math.pi * i / segs
                pts.append(p + r * (math.cos(a) * u + math.sin(a) * w))
        faces = [[i, (i + 1) % segs, segs + (i + 1) % segs, segs + i] for i in range(segs)]
        if caps:
            faces.append(list(reversed(range(segs))))
            faces.append([segs + i for i in range(segs)])
        self._add(pts, faces, mat)

    def quad(self, a, b, c, d, mat=0, uv=None):
        self._add([a, b, c, d], [[0, 1, 2, 3]], mat, [uv] if uv else None)

    def build(self, name, mats, smooth=False, collection=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        if any(u is not None for u in self.uv):
            layer = me.uv_layers.new(name="UVMap")
            data = []
            for f, u in zip(self.f, self.uv):
                if u is None:
                    data.extend([(0.0, 0.0)] * len(f))
                else:
                    data.extend(u)
            layer.data.foreach_set("uv", np.array(data, np.float32).reshape(-1))
        me.validate()
        for mt in mats:
            me.materials.append(mt)
        me.polygons.foreach_set("material_index", np.array(self.m, np.int32))
        if smooth:
            me.polygons.foreach_set("use_smooth", np.ones(len(self.f), bool))
        obj = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(obj)
        return obj


def _grid(name, x0, x1, y0, y1, step, z=0.0, collection=None):
    nx = max(1, int(round((x1 - x0) / step)))
    ny = max(1, int(round((y1 - y0) / step)))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    verts = [(x, y, z) for y in ys for x in xs]
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append([a, a + 1, a + nx + 2, a + nx + 1])
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj, np.array(verts)


# -------------------------------------------------------------------- grass
def grass_curves(name, x0, x1, y0, y1, density_fn, per_m2, length, dry=0.0, seed=1, collection=None,
                 radius=0.0014, npts=4, tilt=0.35, len_jitter=1.1, clump=0.0):
    """Static grass blades as a hair Curves object (no particle re-evaluation per frame)."""
    rng = np.random.default_rng(seed)
    area = (x1 - x0) * (y1 - y0)
    n_try = int(area * per_m2)
    xs = rng.uniform(x0, x1, n_try)
    ys = rng.uniform(y0, y1, n_try)
    keep = rng.uniform(0, 1, n_try) < np.clip(density_fn(xs, ys), 0, 1)
    xs, ys = xs[keep], ys[keep]
    if clump > 0:
        # pull blades towards clump centres for a tufty look
        cx = np.round(xs / clump) * clump + rng.normal(0, clump * 0.15, len(xs))
        cy = np.round(ys / clump) * clump + rng.normal(0, clump * 0.15, len(ys))
        xs = xs + (cx - xs) * 0.55
        ys = ys + (cy - ys) * 0.55
    n = len(xs)
    L = length * (0.4 + len_jitter * rng.uniform(0, 1, n) ** 0.9)
    t0 = np.abs(rng.normal(0, tilt, n))
    az = rng.uniform(0, 2 * np.pi, n)
    bend = rng.uniform(0.1, 0.9, n)
    s = np.linspace(0, 1, npts)
    ang = t0[:, None] + bend[:, None] * s[None, :] ** 1.5
    ds = 1.0 / (npts - 1)
    horiz = np.cumsum(np.sin(ang) * ds, axis=1) - np.sin(ang[:, :1]) * ds
    vert = np.cumsum(np.cos(ang) * ds, axis=1) - np.cos(ang[:, :1]) * ds
    pos = np.empty((n, npts, 3), np.float32)
    pos[:, :, 0] = xs[:, None] + np.cos(az)[:, None] * horiz * L[:, None]
    pos[:, :, 1] = ys[:, None] + np.sin(az)[:, None] * horiz * L[:, None]
    pos[:, :, 2] = vert * L[:, None] - 0.01
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves([npts] * n)
    cv.position_data.foreach_set("vector", pos.reshape(-1))
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    r = (radius * (1.0 - 0.85 * s))[None, :].repeat(n, 0).astype(np.float32)
    rad.data.foreach_set("value", r.reshape(-1))
    cv.materials.append(M.grass_material("Grass" + ("Dry" if dry > 0.3 else ""), dry=dry))
    obj = bpy.data.objects.new(name, cv)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj, n


# ------------------------------------------------------------------- fence
def build_fence(coll):
    concrete = M.concrete_material("FenceConcrete", tone=0.3, stains=0.8, scale=2.0)
    steel = M.metal_material("FenceSteel")
    mesh = M.chainlink_material()
    mb = MeshBuilder()
    x0, x1 = FENCE_X
    mb.box(((x0 + x1) / 2, 0.0, FENCE_PLINTH / 2), (x1 - x0, 0.32, FENCE_PLINTH), 0)
    xs = np.arange(x0, x1 + 0.01, POST_SPACING)
    for x in xs:
        mb.cone((x, 0, FENCE_PLINTH - 0.05), (x, 0, FENCE_TOP + 0.03), 0.038, 0.038, 1, segs=12)
        mb.cone((x, 0, FENCE_TOP + 0.03), (x, 0, FENCE_TOP + 0.07), 0.045, 0.02, 1, segs=12)
    # top and bottom rails
    mb.cone((x0, 0, FENCE_TOP), (x1, 0, FENCE_TOP), 0.021, 0.021, 1, segs=12, caps=False)
    mb.cone((x0, 0.0, FENCE_PLINTH + 0.06), (x1, 0.0, FENCE_PLINTH + 0.06), 0.008, 0.008, 1, segs=6, caps=False)
    # chain-link mesh, slightly on the prison side of the posts
    yz = 0.042
    z0, z1 = FENCE_PLINTH + 0.03, FENCE_TOP - 0.02
    for xa, xb in zip(xs[:-1], xs[1:]):
        mb.quad((xa, yz, z0), (xb, yz, z0), (xb, yz, z1), (xa, yz, z1), 2,
                uv=[(xa, z0), (xb, z0), (xb, z1), (xa, z1)])
    obj = mb.build("Fence", [concrete, steel, mesh], collection=coll)
    for p in obj.data.polygons:
        p.use_smooth = p.material_index == 1
    return obj


# ---------------------------------------------------------------- building
def build_cellblock(coll, x0=-46.0, x1=46.0, y0=BUILDING_Y, depth=15.0, floors=7, floor_h=3.25, name="CellBlock",
                    bay=5.0, tone=0.2):
    conc = M.concrete_material(name + "Concrete", tone=tone, stains=1.0)
    conc_dark = M.concrete_material(name + "ConcreteDark", tone=tone * 0.7, stains=1.2)
    glass = M.glass_window_material()
    steel = M.metal_material("WindowBars", color=(0.08, 0.08, 0.085), rough=0.5, rust=0.4)
    mb = MeshBuilder()
    H = floors * floor_h
    base_h = 1.2
    # core volume (back, sides, roof)
    y1 = y0 + depth
    mb.box(((x0 + x1) / 2, y0 + depth / 2 + 0.25, H / 2), (x1 - x0, depth - 0.5, H), 0)
    # facade with window openings: built bay by bay as a pierced wall
    win_w, win_h = 0.62, 1.35
    pil_w, pil_d = 0.95, 0.55
    nb = int(round((x1 - x0) / bay))
    xs = np.linspace(x0, x1, nb + 1)
    yf = y0  # facade plane
    rec = 0.22  # window recess depth
    for i in range(nb):
        bx0, bx1 = xs[i] + pil_w / 2, xs[i + 1] - pil_w / 2
        cx = (bx0 + bx1) / 2
        wins = [(cx - 0.75 - win_w / 2, cx - 0.75 + win_w / 2), (cx + 0.75 - win_w / 2, cx + 0.75 + win_w / 2)]
        for fl in range(floors):
            z0 = fl * floor_h
            z1 = z0 + floor_h
            if fl == 0:
                mb.quad((bx0, yf, z0), (bx1, yf, z0), (bx1, yf, z1), (bx0, yf, z1), 1)
                continue
            wz0 = z0 + 0.95
            wz1 = wz0 + win_h
            mb.quad((bx0, yf, z0), (bx1, yf, z0), (bx1, yf, wz0), (bx0, yf, wz0), 0)
            mb.quad((bx0, yf, wz1), (bx1, yf, wz1), (bx1, yf, z1), (bx0, yf, z1), 0)
            edges = [bx0] + [v for w in wins for v in w] + [bx1]
            for k in range(0, len(edges), 2):
                a, b = edges[k], edges[k + 1]
                mb.quad((a, yf, wz0), (b, yf, wz0), (b, yf, wz1), (a, yf, wz1), 0)
            for a, b in wins:
                # reveals
                mb.quad((a, yf, wz0), (a, yf + rec, wz0), (a, yf + rec, wz1), (a, yf, wz1), 0)
                mb.quad((b, yf + rec, wz0), (b, yf, wz0), (b, yf, wz1), (b, yf + rec, wz1), 0)
                mb.quad((a, yf + rec, wz0), (a, yf, wz0), (b, yf, wz0), (b, yf + rec, wz0), 0)
                mb.quad((a, yf, wz1), (a, yf + rec, wz1), (b, yf + rec, wz1), (b, yf, wz1), 0)
                mb.quad((a, yf + rec, wz0), (b, yf + rec, wz0), (b, yf + rec, wz1), (a, yf + rec, wz1), 2)
                for k in range(1, 4):
                    xb = a + (b - a) * k / 4
                    mb.box((xb, yf + 0.06, (wz0 + wz1) / 2), (0.025, 0.025, win_h), 3)
                # sill
                mb.box(((a + b) / 2, yf - 0.06, wz0 - 0.04), (b - a + 0.16, 0.14, 0.08), 0)
        # floor ledges across the bay
        for fl in range(1, floors):
            mb.box((cx, yf - 0.06, fl * floor_h), (bx1 - bx0, 0.12, 0.16), 0)
    # pilasters (piers) and a heavy cornice
    for x in xs:
        mb.box((x, yf - pil_d / 2 + 0.02, H / 2 + 0.3), (pil_w, pil_d, H + 0.6), 0)
    mb.box(((x0 + x1) / 2, yf - 0.35, H + 0.45), (x1 - x0 + 1.2, 0.9, 0.9), 0)
    mb.box(((x0 + x1) / 2, yf - 0.1, base_h / 2), (x1 - x0, 0.3, base_h), 1)
    # roof clutter
    rnd = random.Random(7)
    for k in range(9):
        rx = rnd.uniform(x0 + 6, x1 - 6)
        mb.box((rx, y0 + rnd.uniform(4, depth - 4), H + 1.2), (rnd.uniform(2, 5), rnd.uniform(2, 4), rnd.uniform(1.2, 2.4)), 1)
    mb.box(((x0 + x1) / 2 + 12, y0 + 6, H + 2.0), (6, 5, 4.0), 0)
    obj = mb.build(name, [conc, conc_dark, glass, steel], collection=coll)
    return obj


def build_wing(coll):
    """A lower administration wing on the left framing the yard."""
    return build_cellblock(coll, x0=-78.0, x1=-46.0, y0=30.0, depth=12.0, floors=4, floor_h=3.3, name="AdminWing",
                           bay=4.0, tone=0.26)


def build_guard_tower(coll, x=-31.0, y=4.0):
    conc = M.concrete_material("TowerConcrete", tone=0.26, stains=1.0, scale=2.0)
    steel = M.metal_material("TowerSteel", color=(0.18, 0.18, 0.19), rough=0.5, rust=0.4)
    glass = M.glass_window_material()
    mb = MeshBuilder()
    h = 9.0
    mb.cone((x, y, 0), (x, y, h), 1.25, 1.0, 0, segs=16)
    mb.box((x, y, h + 0.15), (4.2, 4.2, 0.3), 0)
    mb.box((x, y, h + 1.4), (3.6, 3.6, 2.2), 2)
    for dx in (-1.8, 1.8):
        for dy in (-1.8, 1.8):
            mb.box((x + dx, y + dy, h + 1.4), (0.18, 0.18, 2.2), 1)
    mb.box((x, y, h + 2.65), (4.6, 4.6, 0.3), 1)
    mb.box((x, y, h + 0.75), (4.2, 4.2, 0.9), 1)
    # searchlight
    mb.cone((x - 1.6, y - 2.0, h + 3.0), (x - 1.6, y - 2.5, h + 2.85), 0.25, 0.3, 1, segs=12)
    return mb.build("GuardTower", [conc, steel, glass], collection=coll)


def build_light_poles(coll, positions):
    steel = M.metal_material("PoleSteel", color=(0.3, 0.31, 0.32), rough=0.5, rust=0.2)
    lamp = M.simple_material("LampLens", (0.8, 0.8, 0.75), rough=0.2)
    mb = MeshBuilder()
    for x, y in positions:
        mb.cone((x, y, 0), (x, y, 11.0), 0.13, 0.08, 0, segs=10)
        mb.box((x, y - 0.5, 11.0), (1.4, 0.15, 0.15), 0)
        for dx in (-0.5, 0.5):
            mb.box((x + dx, y - 0.65, 10.85), (0.45, 0.25, 0.3), 1)
    return mb.build("LightPoles", [steel, lamp], collection=coll)


def build_barriers(coll):
    """Concrete jersey barriers along the patrol road outside the fence."""
    conc = M.concrete_material("BarrierConcrete", tone=0.34, stains=0.7, scale=3.0)
    mb = MeshBuilder()
    for x in np.arange(-40, 41, 3.8):
        if -6 < x < 8:
            continue  # gap where he crosses
        y = -6.2
        prof = [(-0.3, 0.0), (0.3, 0.0), (0.25, 0.08), (0.1, 0.32), (0.08, 0.81), (-0.08, 0.81), (-0.1, 0.32), (-0.25, 0.08)]
        verts = [(x - 1.8, y + p[0], p[1]) for p in prof] + [(x + 1.8, y + p[0], p[1]) for p in prof]
        n = len(prof)
        faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        faces.append(list(reversed(range(n))))
        faces.append([n + i for i in range(n)])
        mb._add(verts, faces, 0)
    return mb.build("Barriers", [conc], collection=coll)


# -------------------------------------------------------------------- trees
def _foliage_nodes(leaf_obj, density, name):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes
    L = ng.links
    gin = N.new("NodeGroupInput")
    gout = N.new("NodeGroupOutput")
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.distribute_method = "RANDOM"
    dist.inputs["Density"].default_value = density
    L.new(gin.outputs[0], dist.inputs["Mesh"])
    # push points into the crown volume
    rnd = N.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT"
    rnd.inputs["Min"].default_value = 0.0
    rnd.inputs["Max"].default_value = 0.9
    scale = N.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    L.new(dist.outputs["Normal"], scale.inputs[0])
    L.new(rnd.outputs["Value"], scale.inputs["Scale"])
    neg = N.new("ShaderNodeVectorMath")
    neg.operation = "SCALE"
    L.new(scale.outputs[0], neg.inputs[0])
    neg.inputs["Scale"].default_value = -0.9
    setp = N.new("GeometryNodeSetPosition")
    L.new(dist.outputs["Points"], setp.inputs["Geometry"])
    L.new(neg.outputs[0], setp.inputs["Offset"])
    oi = N.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = leaf_obj
    rrot = N.new("FunctionNodeRandomValue")
    rrot.data_type = "FLOAT_VECTOR"
    rrot.inputs["Min"].default_value = (0, 0, 0)
    rrot.inputs["Max"].default_value = (6.28, 6.28, 6.28)
    rsc = N.new("FunctionNodeRandomValue")
    rsc.data_type = "FLOAT"
    rsc.inputs["Min"].default_value = 1.0
    rsc.inputs["Max"].default_value = 1.7
    inst = N.new("GeometryNodeInstanceOnPoints")
    L.new(setp.outputs[0], inst.inputs["Points"])
    L.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    L.new(rrot.outputs["Value"], inst.inputs["Rotation"])
    L.new(rsc.outputs["Value"], inst.inputs["Scale"])
    # one mesh (one BVH) renders much faster than ~10^5 tiny instances
    real = N.new("GeometryNodeRealizeInstances")
    L.new(inst.outputs[0], real.inputs[0])
    L.new(real.outputs[0], gout.inputs[0])
    return ng


def _leaf_object(coll):
    leaf = bpy.data.objects.get("LeafProto")
    if leaf:
        return leaf
    w, l = 0.05, 0.085
    verts = [(0, 0, 0), (w * 0.5, l * 0.35, 0.004), (w * 0.35, l * 0.75, 0.006), (0, l, 0.0), (-w * 0.35, l * 0.75, 0.006), (-w * 0.5, l * 0.35, 0.004)]
    me = bpy.data.meshes.new("LeafProto")
    me.from_pydata(verts, [], [[0, 1, 2, 3, 4, 5]])
    me.materials.append(M.leaf_material())
    leaf = bpy.data.objects.new("LeafProto", me)
    coll.objects.link(leaf)
    leaf.hide_render = True
    leaf.hide_viewport = True
    return leaf


def build_tree(coll, x, y, height=8.0, seed=0, crown=1.0, name="Tree"):
    rnd = random.Random(seed)
    bark = M.bark_material()
    mb = MeshBuilder()
    clusters = []

    def branch(p0, d, length, r0, depth):
        p1 = p0 + d * length
        r1 = r0 * 0.68
        mb.cone(p0, p1, r0, r1, 0, segs=8 if depth < 2 else 6)
        if depth >= 3 or length < 0.6:
            clusters.append((p1, 0.9 * crown + 0.25 * rnd.random()))
            return
        n = rnd.choice([2, 3, 3])
        for _ in range(n):
            nd = d + np.array([rnd.uniform(-0.9, 0.9), rnd.uniform(-0.9, 0.9), rnd.uniform(0.0, 0.7)])
            nd /= np.linalg.norm(nd)
            branch(p1, nd, length * rnd.uniform(0.6, 0.8), r1, depth + 1)
        if depth >= 1:
            clusters.append((p1, 0.75 * crown))

    trunk_h = height * 0.38
    base = np.array([x, y, -0.1])
    branch(base, np.array([0, 0, 1.0]), trunk_h, 0.16 * height / 8, 0)
    trunk = mb.build(name + "Wood", [bark], smooth=True, collection=coll)
    # crown emitter (hidden), leaves instanced on its surface by geometry nodes
    cm = MeshBuilder()
    for c, r in clusters:
        _icosphere(cm, c + np.array([0, 0, 0.3]), r * rnd.uniform(1.1, 1.5))
    crown_obj = cm.build(name + "Crown", [M.leaf_material()], collection=coll)
    leaf = _leaf_object(coll)
    mod = crown_obj.modifiers.new("Leaves", "NODES")
    mod.node_group = _foliage_nodes(leaf, 105.0, name + "Foliage")
    return trunk, crown_obj


def _icosphere(mb, c, r, sub=1):
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v) / np.linalg.norm(v) for v in verts]
    for _ in range(sub):
        cache = {}
        nf = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c_ in faces:
            ab, bc, ca = mid(a, b), mid(b, c_), mid(c_, a)
            nf += [(a, ab, ca), (b, bc, ab), (c_, ca, bc), (ab, bc, ca)]
        faces = nf
    rnd = np.random.default_rng(int(abs(c[0] * 100 + c[1] * 10)))
    pts = [np.array(c) + v * r * (1 + 0.18 * rnd.standard_normal()) * np.array([1, 1, 0.8]) for v in verts]
    mb._add(pts, [list(f) for f in faces], 0)


def build_treeline(coll, y, x0, x1, seed=3, scale=1.0, name="Treeline"):
    """Cheap distant trees: lumpy blobs with leaf shading (no instancing)."""
    rnd = random.Random(seed)
    mb = MeshBuilder()
    x = x0
    while x < x1:
        h = rnd.uniform(7, 13) * scale
        r = rnd.uniform(3, 5.5) * scale
        for k in range(3):
            _icosphere(mb, np.array([x + rnd.uniform(-1.5, 1.5), y + rnd.uniform(-3, 3), h - k * r * 0.6]), r * rnd.uniform(0.7, 1.0), sub=2)
        x += rnd.uniform(4, 8) * scale
    mat = M.leaf_material("DistantLeaves", hue=-0.02)
    obj = mb.build(name, [mat], smooth=True, collection=coll)
    tex = bpy.data.textures.new(name + "Lumps", "CLOUDS")
    tex.noise_scale = 1.2
    d = obj.modifiers.new("Lumps", "DISPLACE")
    d.texture = tex
    d.strength = 1.2
    return obj


# ---------------------------------------------------------------------- sky
SUN_ELEV = 34.0
SUN_AZIM = -62.0  # degrees from -Y towards -X: sun from the front-left of the cameras


def sun_direction():
    e, a = math.radians(SUN_ELEV), math.radians(SUN_AZIM)
    # vector pointing from the ground towards the sun
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


SKY_STRENGTH = 0.18
SUN_ENERGY = 3.6


def build_sky(scene, strength=None, sun_energy=None):
    w = bpy.data.worlds.new("Sky")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    g = M.G(nt)
    d = sun_direction()

    def sky_node(vec=None):
        sk = g.n("ShaderNodeTexSky")
        sk.sky_type = "MULTIPLE_SCATTERING"
        sk.sun_disc = False
        sk.sun_elevation = math.radians(SUN_ELEV)
        sk.sun_rotation = math.atan2(d[0], d[1]) % (2 * math.pi)
        sk.altitude = 100.0
        sk.aerosol_density = 1.4
        if vec is not None:
            sk.inputs["Vector"].default_value = vec
        return sk

    sky = sky_node()
    # reference radiance of a bright patch of sky, used to expose the clouds
    ref = sky_node((-d[0] * 0.7, -d[1] * 0.7, 0.35))
    lum = g.n("ShaderNodeRGBToBW")
    g.l(ref.outputs[0], lum.inputs[0])
    tc = g.n("ShaderNodeTexCoord")
    sep = g.n("ShaderNodeSeparateXYZ")
    g.l(tc.outputs["Generated"], sep.inputs[0])
    zc = g.math("ADD", g.math("MAXIMUM", sep.outputs["Z"], 0.0), 0.07)
    comb = g.n("ShaderNodeCombineXYZ")
    g.l(g.math("DIVIDE", sep.outputs["X"], zc), comb.inputs[0])
    g.l(g.math("DIVIDE", sep.outputs["Y"], zc), comb.inputs[1])
    noise = g.n("ShaderNodeTexNoise", inputs={"Scale": 0.5, "Detail": 7.0, "Roughness": 0.6, "Distortion": 0.3})
    g.l(comb.outputs[0], noise.inputs["Vector"])
    dens = g.ramp(noise.outputs["Fac"], [(0.47, (0, 0, 0)), (0.6, (1, 1, 1))])
    horizon = g.math("SUBTRACT", 1.0, g.math("POWER", g.math("SUBTRACT", 1.0, g.math("MAXIMUM", sep.outputs["Z"], 0.0)), 8.0))
    cover = g.math("MULTIPLY", dens.outputs[0], g.math("MINIMUM", g.math("MULTIPLY", horizon, 2.5), 1.0))
    detail = g.n("ShaderNodeTexNoise", inputs={"Scale": 2.4, "Detail": 6.0})
    g.l(comb.outputs[0], detail.inputs["Vector"])
    shade = g.ramp(detail.outputs["Fac"], [(0.25, (0.5, 0.52, 0.58)), (0.75, (1.0, 0.99, 0.96))])
    cloud = g.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    cloud.inputs[0].default_value = 1.0
    g.l(shade.outputs[0], cloud.inputs[6])
    gain = g.n("ShaderNodeCombineXYZ")
    k = g.math("MULTIPLY", lum.outputs[0], 3.2)
    for i in range(3):
        g.l(k, gain.inputs[i])
    g.l(gain.outputs[0], cloud.inputs[7])
    col = g.mix(cover, sky.outputs[0], cloud.outputs[2])
    bg = g.n("ShaderNodeBackground")
    g.l(col, bg.inputs[0])
    bg.inputs[1].default_value = SKY_STRENGTH if strength is None else strength
    out = g.n("ShaderNodeOutputWorld")
    g.l(bg.outputs[0], out.inputs[0])

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = SUN_ENERGY if sun_energy is None else sun_energy
    sun.data.angle = math.radians(0.8)
    sun.data.color = (1.0, 0.94, 0.86)
    sun.rotation_euler = _look_rot(-d)
    scene.collection.objects.link(sun)
    return w, sun


def _look_rot(direction):
    from mathutils import Vector

    return Vector(direction).to_track_quat("-Z", "Y").to_euler()


# -------------------------------------------------------------------- build
def build(scene):
    coll = bpy.data.collections.new("Prison")
    scene.collection.children.link(coll)
    build_sky(scene)

    # base ground (very large), patrol road outside the fence
    mb = MeshBuilder()
    mb.quad((-400, -400, 0), (400, -400, 0), (400, 400, 0), (-400, 400, 0), 0)
    ground = mb.build("Ground", [M.ground_material("GroundFar", dry=0.25)], collection=coll)
    mb = MeshBuilder()
    mb.quad((-80, -5.4, 0.004), (80, -5.4, 0.004), (80, -1.9, 0.004), (-80, -1.9, 0.004), 0)
    mb.quad((-80, BUILDING_Y - 4.0, 0.004), (60, BUILDING_Y - 4.0, 0.004), (60, BUILDING_Y - 1.0, 0.004), (-80, BUILDING_Y - 1.0, 0.004), 1)
    mb.build("Roads", [M.dirt_road_material(), M.concrete_material("Walkway", tone=0.5, stains=0.3, scale=4.0)], collection=coll)

    build_fence(coll)
    build_cellblock(coll)
    build_wing(coll)
    build_guard_tower(coll)
    build_light_poles(coll, [(-18.0, 2.2), (15.0, 2.2), (40.0, 2.2)])
    build_barriers(coll)

    # trees in and around the yard
    build_tree(coll, 7.5, 33.0, height=8.5, seed=11, crown=1.1, name="TreeA")
    build_tree(coll, -21.0, 28.0, height=7.0, seed=5, crown=1.0, name="TreeB")
    build_tree(coll, 27.0, 24.0, height=9.0, seed=8, crown=1.15, name="TreeC")
    build_tree(coll, -12.0, -14.0, height=7.5, seed=21, crown=1.0, name="TreeD")
    build_treeline(coll, 95.0, -160, 160, seed=4, scale=1.4, name="TreelineBack")
    build_treeline(coll, 40.0, 62, 160, seed=9, scale=1.0, name="TreelineRight")
    build_treeline(coll, 15.0, -160, -82, seed=12, scale=1.0, name="TreelineLeft")

    # grass: dense where cameras get close, sparse further away
    def yard_density(x, y):
        near = (y < 6.5) & (x > -38) & (x < 8)
        mid = (y < 16) & (x > -45) & (x < 30)
        return np.where(near, 1.0, np.where(mid, 0.14, 0.03))

    _, n1 = grass_curves("YardGrass", -45, 30, 0.17, 40.0, yard_density, 1300, 0.10, dry=0.0, seed=3, collection=coll,
                         radius=0.0019)

    def outside_density(x, y):
        road = (y < -1.9) & (y > -5.45)
        band = (y > -1.9) & (x > -24) & (x < 8)
        core = (x > -6) & (x < 10) & (y > -38)
        return np.where(road, 0.0, np.where(band | core, 1.0, 0.07))

    _, n2 = grass_curves("OutsideGrass", -26, 26, -38.0, -0.17, outside_density, 900, 0.15, dry=0.45, seed=4,
                         collection=coll, radius=0.0021)

    def weed_density(x, y):
        band = (y > -1.9) & (y < -0.17) & (x > -24) & (x < 10)
        nearbar = (np.abs(y + 6.2) < 0.9) & (x > -30) & (x < 30)
        return np.where(band | nearbar, 1.0, 0.0)

    _, n3 = grass_curves("Weeds", -30, 30, -7.2, -0.17, weed_density, 120, 0.42, dry=0.75, seed=9, collection=coll,
                         radius=0.0022, npts=5, tilt=0.25, clump=0.35)
    print("grass strands: %d yard, %d outside, %d weeds" % (n1, n2, n3))
    return coll
