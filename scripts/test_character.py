"""Quick look-dev render of the prisoner in his rest pose (front / side / head)."""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402

from escape import character  # noqa: E402

out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/tmp/char"
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
t = time.time()
info = character.build_character()
print("character built in %.1fs, height %.3f" % (time.time() - t, info["height"]))

sc.render.engine = "CYCLES"
sc.cycles.samples = 24
sc.cycles.use_denoising = True
sc.render.resolution_x = 540
sc.render.resolution_y = 720
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.45, 0.55, 0.7, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
sc.collection.objects.link(sun)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
views = {
    "front": ((0, -4.2, 1.0), (math.radians(90), 0, 0), 50),
    "side": ((4.2, 0, 1.0), (math.radians(90), 0, math.radians(90)), 50),
    "back": ((0, 4.2, 1.0), (math.radians(90), 0, math.radians(180)), 50),
    "head": ((0.22, -0.7, 1.66), (math.radians(90), 0, math.radians(17)), 85),
    "feet": ((0.6, -1.3, 0.35), (math.radians(78), 0, math.radians(25)), 50),
}
which = [a for a in sys.argv if a in views] or list(views)
for name in which:
    loc, rot, lens = views[name]
    cam.location = loc
    cam.rotation_euler = rot
    cam.data.lens = lens
    sc.render.filepath = os.path.join(out, name + ".png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(name, "%.1fs" % (time.time() - t))
if "--save" in sys.argv:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "char.blend"))
