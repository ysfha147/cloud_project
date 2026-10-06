"""Debug the run cycle: stick-figure plots + optional Cycles frames."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from escape import character, rigmath, locomotion  # noqa: E402

out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 5.0
bpy.ops.wm.read_factory_settings(use_empty=True)
info = character.build_character()
rig = info["rig"]
sk = rigmath.Skeleton(rig)
body = locomotion.BodyInfo(sk)
run = locomotion.Run(body, [(0, 0), (40, 0)], 0.0, 3.0, [(0.0, speed), (3.0, speed)])

# rest check: solving the rest pose must give identity
rest_c = dict(pelvis_pos=sk.head["pelvis"], pelvis_rot=np.eye(3), spine_rot=np.eye(3), head_rot=np.eye(3))
for s in "lr":
    L = sk.limb["arm_" + s]
    rest_c["hand_" + s] = sk.head["hand_" + s]
    rest_c["elbow_" + s] = sk.head["lowerarm_" + s] - (sk.head["upperarm_" + s] + sk.head["hand_" + s]) / 2
    rest_c["foot_" + s] = sk.head["foot_" + s]
    rest_c["foot_rot_" + s] = np.eye(3)
    rest_c["knee_" + s] = sk.head["calf_" + s] - (sk.head["thigh_" + s] + sk.head["foot_" + s]) / 2
    rest_c["grip_" + s] = -0.06 / 1.25
P = rigmath.solve(sk, rest_c)
err = max(np.abs(P.basis(n) - np.eye(4)).max() for n in sk.order if "thumb" not in n and "index" not in n and "middle" not in n and "ring" not in n and "pinky" not in n)
print("rest pose max basis error (non-finger): %.2e" % err)

fps = 24
frames = list(range(0, 24))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

chains = [["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head"]]
for s in "lr":
    chains.append(["spine_03", "clavicle_" + s, "upperarm_" + s, "lowerarm_" + s, "hand_" + s])
    chains.append(["pelvis", "thigh_" + s, "calf_" + s, "foot_" + s, "ball_" + s])
nf = len(frames)
fig, axes = plt.subplots(2, nf // 2, figsize=(20, 8))
poses = []
for idx, f in enumerate(frames):
    t = f / fps
    c = run.controllers(t)
    P = rigmath.solve(sk, c)
    poses.append((t, c, P))
for idx, (t, c, P) in enumerate(poses[::2]):
    for row, (i, j) in enumerate(((0, 2), (1, 2))):
        ax = axes[row][idx]
        cx = P.joint("pelvis")[i]
        for ch in chains:
            pts = np.array([P.joint(n) for n in ch] + [P.M[ch[-1]][:3, 3] + P.M[ch[-1]][:3, 1] * sk.len[ch[-1]]])
            col = "r" if ch[-1].endswith("_l") else ("b" if ch[-1].endswith("_r") else "k")
            ax.plot(pts[:, i] - cx, pts[:, j], col + "-", lw=1.5)
        for sd in "lr":
            a = c["foot_" + sd]
            ax.plot([a[i] - cx], [a[j]], "o", color="r" if sd == "l" else "b", ms=4, mfc="none")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.05, 1.9)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title("t=%.2f" % t, fontsize=8)
        ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(out, "run_stick.png"), dpi=70)
t, c, P = poses[0]
for sd in "lr":
    print(sd, "hip", P.joint("thigh_" + sd).round(3), "knee", P.joint("calf_" + sd).round(3), "ankle tgt", c["foot_" + sd].round(3), "got", P.joint("foot_" + sd).round(3), "pole", c["knee_" + sd].round(2))

# reach / IK error stats: distance between requested ankle and achieved ankle
errs = []
for t, c, P in poses:
    for s in "lr":
        errs.append(np.linalg.norm(P.joint("foot_" + s) - c["foot_" + s]))
print("max ankle IK miss: %.3f m" % max(errs))

if "--render" in sys.argv:
    from escape import render_settings
    from mathutils import Vector

    sc = bpy.context.scene
    allp = [rigmath.solve(sk, run.controllers(f / fps)) for f in range(0, 49)]
    rigmath.bake(rig, sk, allp, frame_start=1)
    render_settings.apply(sc, width=480, height=480, samples=8, motion_blur=False, preview=True)
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sc.collection.objects.link(sun)
    gp = bpy.data.objects.new("floor", bpy.data.meshes.new("floor"))
    gp.data.from_pydata([(-5, -5, 0), (60, -5, 0), (60, 5, 0), (-5, 5, 0)], [], [[0, 1, 2, 3]])
    sc.collection.objects.link(gp)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = 50
    for f in [int(x) for x in sys.argv[sys.argv.index("--render") + 1].split(",")]:
        sc.frame_set(f)
        px = rig.pose.bones["pelvis"].head
        hip = rig.matrix_world @ px
        cam.location = (hip.x + 0.8, -5.5, 1.1)
        cam.rotation_euler = (Vector((hip.x, hip.y, 0.95)) - Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(out, "run_%03d.png" % f)
        bpy.ops.render.render(write_still=True)
