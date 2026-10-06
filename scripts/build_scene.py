"""Build the whole prison-escape scene and save it as a .blend file.

Usage:
  python scripts/build_scene.py --out build/escape.blend [--lite] [--tend 17.5]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402

from escape import bake, cameras, character, environment, render_settings  # noqa: E402
from escape import materials as M  # noqa: E402

args = sys.argv
out = args[args.index("--out") + 1]
t_end = float(args[args.index("--tend") + 1]) if "--tend" in args else 17.0
lite = "--lite" in args

t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
if lite:
    coll = bpy.data.collections.new("Prison")
    sc.collection.children.link(coll)
    environment.build_sky(sc)
    environment.build_fence(coll)
    environment.build_cellblock(coll)
    environment.build_wing(coll)
    environment.build_guard_tower(coll)
    environment.build_barriers(coll)
    mb = environment.MeshBuilder()
    mb.quad((-400, -400, 0), (400, -400, 0), (400, 400, 0), (-400, 400, 0), 0)
    mb.build("Ground", [M.ground_material("GroundFar", dry=0.25)], collection=coll)
else:
    environment.build(sc)
info = character.build_character()
if lite:
    for m in info["body"].modifiers:
        if m.type == "PARTICLE_SYSTEM":
            m.show_render = False
perf, sk, poses, ctrls = bake.bake_performance(info["rig"], t_end=t_end)
shots = cameras.build_shots(perf, poses, t_end)
cameras.build_cameras(sc, shots)
for sh in shots:
    print("shot %-13s %6.2f - %6.2f  frames %d-%d" % (sh.name, sh.t0, sh.t1, round(sh.t0 * 24) + 1, round(sh.t1 * 24)))
print("Tc %.2f flight %.2f land %.2f sprint %.2f" % (perf.Tc, perf.flight_t0, perf.t_land, perf.t_sprint))
render_settings.apply(sc)
sc.frame_start = 1
sc.frame_end = int(round(t_end * 24))
# events for the soundtrack
import json  # noqa: E402

ev_path = os.path.splitext(out)[0] + "_events.json"
with open(ev_path, "w") as fh:
    json.dump(dict(events=[list(e) for e in perf.events], Tc=perf.Tc, flight=perf.flight_t0, land=perf.t_land,
                   sprint=perf.t_sprint, cuts=[[s.name, s.t0, s.t1] for s in shots], t_end=t_end), fh, indent=1)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
print("saved %s in %.1fs" % (out, time.time() - t0))
