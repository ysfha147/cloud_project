"""Low-cost animation preview: character + fence only, contact sheets."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from escape import bake, character, environment, render_settings  # noqa: E402
from escape import materials as M  # noqa: E402

args = sys.argv
out = args[args.index("--out") + 1]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
coll = bpy.data.collections.new("Prison")
sc.collection.children.link(coll)
environment.build_sky(sc)
environment.build_fence(coll)
mb = environment.MeshBuilder()
mb.quad((-200, -200, 0), (200, -200, 0), (200, 200, 0), (-200, 200, 0), 0)
mb.build("Ground", [M.ground_material("GroundFar", dry=0.25)], collection=coll)
info = character.build_character()
rig = info["rig"]
for m in info["body"].modifiers:
    if m.type == "PARTICLE_SYSTEM" and "--hair" not in args:
        m.show_render = False
perf, sk, poses, ctrls = bake.bake_performance(rig, t_end=float(args[args.index("--tend") + 1]) if "--tend" in args else 18.0)
print("Tc %.2f  flight %.2f  land %.2f  sprint %.2f" % (perf.Tc, perf.flight_t0, perf.t_land, perf.t_sprint))
res = (args[args.index("--res") + 1] if "--res" in args else "480x270").split("x")
render_settings.apply(sc, width=int(res[0]), height=int(res[1]), samples=6, motion_blur=False, preview=True)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens = 35
mode = args[args.index("--view") + 1] if "--view" in args else "climb"
times = [float(x) for x in args[args.index("--times") + 1].split(",")]
files = []
for t in times:
    f = int(round(t * 24)) + 1
    sc.frame_set(f)
    hip = rig.matrix_world @ rig.pose.bones["pelvis"].head
    if mode == "climb":
        cam.location = (1.4 + 4.5, -2.8, 2.0)
        tgt = Vector((1.4, 0.0, 2.0))
    elif mode == "front":
        cam.location = (1.4 + 0.8, -7.0, 1.8)
        tgt = Vector((1.4, 0.0, 1.8))
    else:  # follow
        cam.location = (hip.x + 2.5, hip.y - 5.0, 1.4)
        tgt = Vector((hip.x, hip.y, 1.0))
    cam.rotation_euler = (tgt - Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()
    fn = os.path.join(out, "f_%04d.png" % f)
    sc.render.filepath = fn
    bpy.ops.render.render(write_still=True)
    files.append(fn)
print("FILES", " ".join(files))
if "--save" in args:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "anim.blend"))
