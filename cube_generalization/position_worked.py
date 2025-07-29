import numpy as np
import random
from isaacsim import SimulationApp

# 0) Isaac Sim init
CONFIG = {
    "width": 1280, "height": 720, "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 0,
    "physics_dt": 1.0/60.0, "rendering_dt": 1.0/30.0,
}
simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
from scipy.spatial.transform import Rotation as R

# — your original local‐point generator —
def generate_grasp_positions(dims_total, approach_offset):
    dims = np.array(dims_total, dtype=float)
    half = dims * 0.5
    face_axes = {
        '+X': (0,1,2), '-X': (0,1,2),
        '+Y': (1,0,2), '-Y': (1,0,2),
        '+Z': (2,0,1), '-Z': (2,0,1),
    }
    fractions = [0.25, 0.5, 0.75]
    pts = []
    for face,(i,j,k) in face_axes.items():
        sign = 1 if face[0]=='+' else -1
        n = np.zeros(3); n[i] = sign
        ej = np.zeros(3); ej[j] = 1
        ek = np.zeros(3); ek[k] = 1
        dj, dk = dims[j], dims[k]
        if dj >= dk:
            long_vec, long_len = ej, dj
        else:
            long_vec, long_len = ek, dk
        base = n * (half[i] + approach_offset)
        for frac in fractions:
            off_long = (frac - 0.5) * long_len
            pts.append(base + long_vec * off_long)
    return pts

# 1) World setup
world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
world.scene.add_default_ground_plane()
world.reset()

# 2) Spawn your box exactly as before
dims_total     = [0.1, 0.2, 0.05]
approach_offset = 0.01
half_z         = dims_total[2] * 0.5

cube = VisualCuboid(
    prim_path="/World/Cube",
    name="my_box",
    position=(0.0, 0.0, float(half_z) + 0.2),
    scale=dims_total,
    color=np.array([0.8,0.8,0.8])
)
world.scene.add(cube)
world.step(render=True)

# 3) Get the 18 correct local points & transform them
local_pts = generate_grasp_positions(dims_total, approach_offset)
pos_w, ori_w = cube.get_world_pose()
rot = R.from_quat([ori_w[1], ori_w[2], ori_w[3], ori_w[0]])

world_pts = [rot.apply(p) + pos_w for p in local_pts]

# 4) Visualize those 18 to be 100% sure they match
from omni.isaac.core.objects import VisualSphere
for i, pw in enumerate(world_pts):
    sphere = VisualSphere(
        prim_path=f"/World/OrigSphere_{i}",
        name=f"orig_{i}",
        position=pw, radius=0.005,
        color=np.array([1.0, 0.0, 0.0])
    )
    world.scene.add(sphere)

# At this point, the red spheres are exactly in the positions your original script made.
# You can now overlay the 15‐grasp assignment on top of these same world_pts[0:15].
# (Continue with the tilt/yaw grouping code, but **use** `world_pts[0:15]` as your contact locations.)

# -- Do your 15‐point grouping & orientation code here, but indexing into world_pts

while simulation_app.is_running():
    world.step(render=True)
    simulation_app.update()

simulation_app.close()
