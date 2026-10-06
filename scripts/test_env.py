"""Environment look-dev: build prison + character, render from a few viewpoints."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from escape import character, environment, render_settings  # noqa: E402

out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
t = time.time()
environment.build(sc)
info = character.build_character()
info["rig"].location = (-8.0, 2.6, 0.0)
print("built in %.1fs" % (time.time() - t))
samples = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 16
render_settings.apply(sc, width=960, height=402, samples=samples, motion_blur=False, preview=True)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam


def look(cam, loc, target, lens):
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens


views = {
    "wide": ((14, -17, 2.2), (-8, 18, 5), 30),
    "fence": ((-8.6, -0.8, 0.55), (-8.0, 4.0, 0.9), 40),
    "outside": ((1.0, -10, 1.3), (0, 0, 2.0), 45),
}
which = [a for a in sys.argv if a in views] or list(views)
for name in which:
    loc, tgt, lens = views[name]
    look(cam, loc, tgt, lens)
    sc.render.filepath = os.path.join(out, name + ".png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(name, "%.1fs" % (time.time() - t))
if "--save" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "env.blend"))
