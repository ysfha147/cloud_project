"""Render a frame range of the saved scene (resumable: skips finished frames).

Usage:
  python scripts/render.py --blend build/escape.blend --out frames/ [--start 1 --end 420]
         [--preset final|animatic] [--every 1]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402

from escape import render_settings  # noqa: E402

args = sys.argv


def arg(name, default=None):
    return args[args.index(name) + 1] if name in args else default


bpy.ops.wm.open_mainfile(filepath=arg("--blend"))
sc = bpy.context.scene
out = arg("--out")
os.makedirs(out, exist_ok=True)
preset = arg("--preset", "final")
if preset == "animatic":
    render_settings.apply(sc, width=480, height=200, samples=4, motion_blur=False, preview=True)
    for o in bpy.data.objects:
        if "Grass" in o.name or "Crown" in o.name:
            o.hide_render = True
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name == "PrisonerBody":
            for m in o.modifiers:
                if m.type == "PARTICLE_SYSTEM":
                    m.show_render = False
elif preset == "preview":
    render_settings.apply(sc, width=960, height=402, samples=12, motion_blur=True, preview=True)
else:
    w, h = (int(x) for x in arg("--res", "1280x536").split("x"))
    render_settings.apply(sc, width=w, height=h, samples=int(arg("--samples", "20")), motion_blur=True)
start = int(arg("--start", sc.frame_start))
end = int(arg("--end", sc.frame_end))
every = int(arg("--every", "1"))
frames = [int(f) for f in arg("--frames").split(",")] if arg("--frames") else range(start, end + 1, every)
for f in frames:
    fn = os.path.join(out, "f_%04d.png" % f)
    if os.path.exists(fn) and "--force" not in args:
        continue
    t = time.time()
    sc.frame_set(f)
    sc.render.filepath = fn
    bpy.ops.render.render(write_still=True)
    print("frame %d %.1fs" % (f, time.time() - t), flush=True)
