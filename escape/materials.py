"""Procedural Cycles materials for the prison-escape scene."""
import math

import bpy


def _new(name):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat, None
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return mat, nt


class G:
    """Tiny helper to wire shader graphs with less noise."""

    def __init__(self, nt):
        self.nt = nt

    def n(self, kind, **props):
        node = self.nt.nodes.new(kind)
        inputs = props.pop("inputs", {})
        for k, v in props.items():
            setattr(node, k, v)
        for k, v in inputs.items():
            node.inputs[k].default_value = v
        return node

    def l(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None, clamp=False):
        m = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.l(v, m.inputs[i])
        return m.outputs[0]

    def vmath(self, op, a, b=None):
        m = self.n("ShaderNodeVectorMath", operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                m.inputs[i].default_value = v
            else:
                self.l(v, m.inputs[i])
        return m

    def mix(self, fac, a, b, blend="MIX"):
        m = self.n("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        for sock, v in ((0, fac), (6, a), (7, b)):
            if isinstance(v, (int, float)):
                m.inputs[sock].default_value = v
            elif isinstance(v, (tuple, list)):
                m.inputs[sock].default_value = v
            else:
                self.l(v, m.inputs[sock])
        return m.outputs[2]

    def ramp(self, fac, stops):
        r = self.n("ShaderNodeValToRGB")
        cr = r.color_ramp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (pos, col) in zip(cr.elements, stops):
            el.position = pos
            el.color = col if len(col) == 4 else (*col, 1.0)
        self.l(fac, r.inputs[0])
        return r

    def out(self, shader):
        o = self.n("ShaderNodeOutputMaterial")
        self.l(shader, o.inputs["Surface"])
        return o


def rest_coords(g):
    """Rest-pose coordinates stored on the mesh (stable under skinning)."""
    a = g.n("ShaderNodeAttribute", attribute_name="rest", attribute_type="GEOMETRY")
    return a.outputs["Vector"]


def skin_material():
    mat, nt = _new("Skin")
    if nt is None:
        return mat
    g = G(nt)
    co = rest_coords(g)
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 18.0, "Detail": 4.0, "Roughness": 0.6})
    g.l(co, n1.inputs["Vector"])
    base = g.ramp(n1.outputs["Fac"], [(0.35, (0.37, 0.205, 0.13)), (0.65, (0.44, 0.255, 0.165))])
    # pores / micro detail
    n2 = g.n("ShaderNodeTexNoise", inputs={"Scale": 900.0, "Detail": 2.0})
    g.l(co, n2.inputs["Vector"])
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.18, "Distance": 0.0004})
    g.l(n2.outputs["Fac"], bump.inputs["Height"])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(base.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.48
    p.inputs["Subsurface Weight"].default_value = 0.0
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.18)
    p.inputs["Subsurface Scale"].default_value = 0.012
    p.inputs["Specular IOR Level"].default_value = 0.45
    g.l(bump.outputs[0], p.inputs["Normal"])
    g.out(p.outputs[0])
    return mat


def eye_material():
    mat, nt = _new("Eye")
    if nt is None:
        return mat
    g = G(nt)
    a = g.n("ShaderNodeAttribute", attribute_name="eyedir", attribute_type="GEOMETRY")
    fwd = g.vmath("DOT_PRODUCT", a.outputs["Vector"], (0.0, -1.0, 0.0))
    d = fwd.outputs["Value"]
    col = g.ramp(d, [(0.80, (0.75, 0.72, 0.68)), (0.83, (0.12, 0.06, 0.03)), (0.95, (0.07, 0.035, 0.02)), (0.965, (0.005, 0.005, 0.005))])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.08
    p.inputs["Coat Weight"].default_value = 0.6
    g.out(p.outputs[0])
    return mat


def hair_material():
    mat, nt = _new("Hair")
    if nt is None:
        return mat
    g = G(nt)
    h = g.n("ShaderNodeBsdfHairPrincipled")
    h.parametrization = "MELANIN"
    h.inputs["Melanin"].default_value = 0.96
    h.inputs["Melanin Redness"].default_value = 0.25
    h.inputs["Roughness"].default_value = 0.38
    h.inputs["Radial Roughness"].default_value = 0.5
    h.inputs["Random Roughness"].default_value = 0.2
    h.inputs["Random Color"].default_value = 0.15
    g.out(h.outputs[0])
    return mat


def uniform_material(name, shirt=True):
    """Khaki prison uniform cotton with weave, seams, wear and dirt."""
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    co = rest_coords(g)
    # large-scale tone variation + dirt
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 6.0, "Detail": 6.0, "Roughness": 0.55})
    g.l(co, n1.inputs["Vector"])
    base = g.ramp(n1.outputs["Fac"], [(0.30, (0.30, 0.215, 0.115)), (0.70, (0.40, 0.30, 0.17))])
    dirt_n = g.n("ShaderNodeTexNoise", inputs={"Scale": 3.0, "Detail": 8.0, "Roughness": 0.7})
    g.l(co, dirt_n.inputs["Vector"])
    dirt = g.ramp(dirt_n.outputs["Fac"], [(0.55, (0, 0, 0)), (0.75, (1, 1, 1))])
    sep = g.n("ShaderNodeSeparateXYZ")
    g.l(co, sep.inputs[0])
    # dirtier towards the hem of the trousers
    low = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.math("MULTIPLY", sep.outputs["Z"], 2.2), clamp=True), 0.8)
    dmask = g.math("MAXIMUM", g.math("MULTIPLY", dirt.outputs[0], 0.5), low)
    col = g.mix(dmask, base.outputs[0], (0.17, 0.13, 0.08, 1.0))
    # woven fabric micro bump
    weave = g.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="DIAGONAL", inputs={"Scale": 420.0, "Distortion": 2.0, "Detail": 1.0})
    g.l(co, weave.inputs["Vector"])
    wr = g.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", inputs={"Scale": 9.0, "Distortion": 9.0, "Detail": 3.0, "Detail Scale": 2.0})
    g.l(co, wr.inputs["Vector"])
    height = g.math("ADD", g.math("MULTIPLY", weave.outputs["Fac"], 0.15), g.math("MULTIPLY", wr.outputs["Fac"], 1.0))
    if shirt:
        # button placket down the front centre + two chest pockets
        ax = g.math("ABSOLUTE", sep.outputs["X"])
        placket = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", ax, 0.012)), 0.0018)
        front = g.math("LESS_THAN", sep.outputs["Y"], -0.05)
        seam = g.math("MULTIPLY", placket, front)
        px = g.math("ABSOLUTE", g.math("SUBTRACT", ax, 0.09))
        pz = g.math("ABSOLUTE", g.math("SUBTRACT", sep.outputs["Z"], 1.33))
        edge_x = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", px, 0.055)), 0.0025)
        edge_z = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", pz, 0.065)), 0.0025)
        inside_x = g.math("LESS_THAN", px, 0.058)
        inside_z = g.math("LESS_THAN", pz, 0.068)
        pocket = g.math("MAXIMUM", g.math("MULTIPLY", edge_x, inside_z), g.math("MULTIPLY", edge_z, inside_x))
        pocket = g.math("MULTIPLY", pocket, front)
        seam = g.math("MAXIMUM", seam, pocket)
        height = g.math("SUBTRACT", height, g.math("MULTIPLY", seam, 1.2))
        col = g.mix(g.math("MULTIPLY", seam, 0.35), col, (0.12, 0.09, 0.05, 1.0))
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.35, "Distance": 0.002})
    g.l(height, bump.inputs["Height"])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.88
    p.inputs["Sheen Weight"].default_value = 0.35
    p.inputs["Sheen Roughness"].default_value = 0.4
    p.inputs["Specular IOR Level"].default_value = 0.25
    g.l(bump.outputs[0], p.inputs["Normal"])
    g.out(p.outputs[0])
    return mat


def shoe_material():
    mat, nt = _new("Shoes")
    if nt is None:
        return mat
    g = G(nt)
    co = rest_coords(g)
    sep = g.n("ShaderNodeSeparateXYZ")
    g.l(co, sep.inputs[0])
    sole = g.math("LESS_THAN", sep.outputs["Z"], 0.02)
    col = g.mix(sole, (0.03, 0.03, 0.035, 1.0), (0.55, 0.53, 0.5, 1.0))
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.55
    g.out(p.outputs[0])
    return mat


# ---------------------------------------------------------------- environment
def concrete_material(name="Concrete", tone=0.36, stains=1.0, scale=1.0):
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    tc = g.n("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 0.35 * scale, "Detail": 6.0, "Roughness": 0.6})
    g.l(co, n1.inputs["Vector"])
    t = tone
    base = g.ramp(n1.outputs["Fac"], [(0.3, (t * 0.82, t * 0.82, t * 0.8)), (0.7, (t * 1.12, t * 1.1, t * 1.05))])
    # vertical rain streaks
    mp = g.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.2 * scale, 2.2 * scale, 0.12 * scale)
    g.l(co, mp.inputs["Vector"])
    n2 = g.n("ShaderNodeTexNoise", inputs={"Scale": 3.0, "Detail": 3.0})
    g.l(mp.outputs[0], n2.inputs["Vector"])
    streak = g.ramp(n2.outputs["Fac"], [(0.5, (0, 0, 0)), (0.72, (1, 1, 1))])
    col = g.mix(g.math("MULTIPLY", streak.outputs[0], 0.55 * stains), base.outputs[0], (t * 0.45, t * 0.43, t * 0.4, 1.0))
    # grime near the ground
    sep = g.n("ShaderNodeSeparateXYZ")
    g.l(tc.outputs["Object"], sep.inputs[0])
    low = g.math("SUBTRACT", 1.0, g.math("MULTIPLY", sep.outputs["Z"], 1.6), clamp=True)
    col = g.mix(g.math("MULTIPLY", low, 0.6 * stains), col, (0.12, 0.11, 0.09, 1.0))
    n3 = g.n("ShaderNodeTexNoise", inputs={"Scale": 40.0 * scale, "Detail": 8.0})
    g.l(co, n3.inputs["Vector"])
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.25, "Distance": 0.01})
    g.l(n3.outputs["Fac"], bump.inputs["Height"])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.9
    g.l(bump.outputs[0], p.inputs["Normal"])
    g.out(p.outputs[0])
    return mat


def metal_material(name="GalvanizedSteel", color=(0.42, 0.43, 0.44), rough=0.42, rust=0.25):
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    tc = g.n("ShaderNodeTexCoord")
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 6.0, "Detail": 8.0, "Roughness": 0.65})
    g.l(tc.outputs["Object"], n1.inputs["Vector"])
    r = g.ramp(n1.outputs["Fac"], [(0.6, (0, 0, 0)), (0.75, (1, 1, 1))])
    col = g.mix(g.math("MULTIPLY", r.outputs[0], rust), (*color, 1.0), (0.22, 0.12, 0.06, 1.0))
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col, p.inputs["Base Color"])
    p.inputs["Metallic"].default_value = 0.85
    p.inputs["Roughness"].default_value = rough
    g.out(p.outputs[0])
    return mat


def chainlink_material(spacing=0.055, wire=0.0032):
    """Diamond chain-link mesh from UVs (u along the fence, v up, in metres)."""
    mat, nt = _new("ChainLink")
    if nt is None:
        return mat
    g = G(nt)
    uv = g.n("ShaderNodeUVMap", uv_map="UVMap")
    sep = g.n("ShaderNodeSeparateXYZ")
    g.l(uv.outputs[0], sep.inputs[0])
    s = spacing * math.sqrt(2.0)
    a = g.math("DIVIDE", g.math("ADD", sep.outputs["X"], sep.outputs["Y"]), s)
    b = g.math("DIVIDE", g.math("SUBTRACT", sep.outputs["X"], sep.outputs["Y"]), s)
    fa = g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", a), 0.5))
    fb = g.math("ABSOLUTE", g.math("SUBTRACT", g.math("FRACT", b), 0.5))
    half = 0.5 - (wire / s)
    da = g.math("SUBTRACT", fa, half)
    db = g.math("SUBTRACT", fb, half)
    wa = g.math("GREATER_THAN", da, 0.0)
    wb = g.math("GREATER_THAN", db, 0.0)
    mask = g.math("MAXIMUM", wa, wb)
    # fake round-wire shading: height peaks at the wire centre
    hgt = g.math("MAXIMUM", g.math("DIVIDE", da, wire / s), g.math("DIVIDE", db, wire / s))
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.8, "Distance": 0.002})
    g.l(hgt, bump.inputs["Height"])
    metal = g.n("ShaderNodeBsdfPrincipled")
    metal.inputs["Base Color"].default_value = (0.2, 0.205, 0.21, 1)
    metal.inputs["Metallic"].default_value = 0.7
    metal.inputs["Roughness"].default_value = 0.5
    g.l(bump.outputs[0], metal.inputs["Normal"])
    tr = g.n("ShaderNodeBsdfTransparent")
    mx = g.n("ShaderNodeMixShader")
    g.l(mask, mx.inputs[0])
    g.l(tr.outputs[0], mx.inputs[1])
    g.l(metal.outputs[0], mx.inputs[2])
    g.out(mx.outputs[0])
    return mat


def glass_window_material():
    mat, nt = _new("WindowGlass")
    if nt is None:
        return mat
    g = G(nt)
    tc = g.n("ShaderNodeTexCoord")
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 0.8, "Detail": 2.0})
    g.l(tc.outputs["Object"], n1.inputs["Vector"])
    col = g.ramp(n1.outputs["Fac"], [(0.3, (0.012, 0.014, 0.017)), (0.7, (0.05, 0.055, 0.06))])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.08
    p.inputs["Specular IOR Level"].default_value = 0.7
    g.out(p.outputs[0])
    return mat


def ground_material(name="Ground", dry=0.0):
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    tc = g.n("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 0.25, "Detail": 6.0, "Roughness": 0.6})
    g.l(co, n1.inputs["Vector"])
    n2 = g.n("ShaderNodeTexNoise", inputs={"Scale": 3.0, "Detail": 8.0, "Roughness": 0.7})
    g.l(co, n2.inputs["Vector"])
    green = (0.035 + 0.05 * dry, 0.065 + 0.02 * dry, 0.012 + 0.01 * dry, 1)
    green2 = (0.06 + 0.07 * dry, 0.085 + 0.02 * dry, 0.02 + 0.012 * dry, 1)
    c = g.ramp(n1.outputs["Fac"], [(0.35, green), (0.65, green2)])
    dirt = g.ramp(n2.outputs["Fac"], [(0.55, (0, 0, 0)), (0.7, (1, 1, 1))])
    col = g.mix(g.math("MULTIPLY", dirt.outputs[0], 0.7), c.outputs[0], (0.11, 0.085, 0.055, 1))
    n3 = g.n("ShaderNodeTexNoise", inputs={"Scale": 60.0, "Detail": 6.0})
    g.l(co, n3.inputs["Vector"])
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.6, "Distance": 0.01})
    g.l(n3.outputs["Fac"], bump.inputs["Height"])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.95
    g.l(bump.outputs[0], p.inputs["Normal"])
    g.out(p.outputs[0])
    return mat


def dirt_road_material():
    mat, nt = _new("DirtRoad")
    if nt is None:
        return mat
    g = G(nt)
    tc = g.n("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 1.5, "Detail": 8.0, "Roughness": 0.65})
    g.l(co, n1.inputs["Vector"])
    col = g.ramp(n1.outputs["Fac"], [(0.3, (0.13, 0.105, 0.075)), (0.7, (0.24, 0.2, 0.15))])
    v = g.n("ShaderNodeTexVoronoi", inputs={"Scale": 90.0})
    g.l(co, v.inputs["Vector"])
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.7, "Distance": 0.008})
    g.l(v.outputs["Distance"], bump.inputs["Height"])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.95
    g.l(bump.outputs[0], p.inputs["Normal"])
    g.out(p.outputs[0])
    return mat


def grass_material(name="Grass", dry=0.0):
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    hi = g.n("ShaderNodeHairInfo")
    tc = g.n("ShaderNodeTexCoord")
    n1 = g.n("ShaderNodeTexNoise", inputs={"Scale": 0.4, "Detail": 3.0})
    g.l(tc.outputs["Generated"], n1.inputs["Vector"])
    c1 = (0.045 + 0.10 * dry, 0.10 + 0.02 * dry, 0.018 + 0.015 * dry, 1)
    c2 = (0.09 + 0.16 * dry, 0.16 + 0.03 * dry, 0.03 + 0.03 * dry, 1)
    c3 = (0.18 + 0.12 * dry, 0.17 + 0.05 * dry, 0.06 + 0.03 * dry, 1)
    r = g.ramp(hi.outputs["Random"], [(0.0, c1), (0.6, c2), (1.0, c3)])
    patch = g.ramp(n1.outputs["Fac"], [(0.4, (0, 0, 0)), (0.65, (1, 1, 1))])
    col = g.mix(g.math("MULTIPLY", patch.outputs[0], 0.5), r.outputs[0], c3)
    root = g.mix(g.math("SUBTRACT", 1.0, hi.outputs["Intercept"]), col, (0.02, 0.03, 0.008, 1), blend="MULTIPLY")
    col = g.mix(0.55, col, root)
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(col, p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.6
    p.inputs["Specular IOR Level"].default_value = 0.3
    tr = g.n("ShaderNodeBsdfTranslucent")
    g.l(col, tr.inputs["Color"])
    mx = g.n("ShaderNodeMixShader")
    mx.inputs[0].default_value = 0.25
    g.l(p.outputs[0], mx.inputs[1])
    g.l(tr.outputs[0], mx.inputs[2])
    g.out(mx.outputs[0])
    return mat


def leaf_material(name="Leaves", hue=0.0):
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    oi = g.n("ShaderNodeObjectInfo")
    c = g.ramp(oi.outputs["Random"], [(0.0, (0.03, 0.07 + hue, 0.012, 1)), (0.5, (0.06, 0.12 + hue, 0.02, 1)), (1.0, (0.12, 0.15 + hue, 0.03, 1))])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(c.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.55
    tr = g.n("ShaderNodeBsdfTranslucent")
    g.l(c.outputs[0], tr.inputs["Color"])
    mx = g.n("ShaderNodeMixShader")
    mx.inputs[0].default_value = 0.3
    g.l(p.outputs[0], mx.inputs[1])
    g.l(tr.outputs[0], mx.inputs[2])
    g.out(mx.outputs[0])
    return mat


def bark_material():
    mat, nt = _new("Bark")
    if nt is None:
        return mat
    g = G(nt)
    tc = g.n("ShaderNodeTexCoord")
    mp = g.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (12, 12, 1.5)
    g.l(tc.outputs["Object"], mp.inputs["Vector"])
    n = g.n("ShaderNodeTexNoise", inputs={"Scale": 4.0, "Detail": 8.0})
    g.l(mp.outputs[0], n.inputs["Vector"])
    c = g.ramp(n.outputs["Fac"], [(0.3, (0.05, 0.04, 0.03)), (0.7, (0.13, 0.11, 0.09))])
    bump = g.n("ShaderNodeBump", inputs={"Strength": 0.8, "Distance": 0.02})
    g.l(n.outputs["Fac"], bump.inputs["Height"])
    p = g.n("ShaderNodeBsdfPrincipled")
    g.l(c.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.9
    g.l(bump.outputs[0], p.inputs["Normal"])
    g.out(p.outputs[0])
    return mat


def simple_material(name, color, rough=0.5, metal=0.0, emission=None, strength=0.0):
    mat, nt = _new(name)
    if nt is None:
        return mat
    g = G(nt)
    p = g.n("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    if emission:
        p.inputs["Emission Color"].default_value = (*emission, 1)
        p.inputs["Emission Strength"].default_value = strength
    g.out(p.outputs[0])
    return mat
