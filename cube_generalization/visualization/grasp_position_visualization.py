"""
One-object visualization of ALL 18 grasp contact positions (3 per face × 6 faces).
- Uses ONLY grasp_generator for dimensions, base orientation, and contact sampling.
- No robots, no IK, no yaw/tilt. Positions only (red spheres).
"""

import numpy as np
from typing import List
from isaacsim import SimulationApp

CONFIG = {
    "width": 1920, "height": 1080, "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": 1<<0 | 1<<3 | 1<<10 | 1<<13 | 1<<14,
    "physics_dt": 1.0/60.0, "rendering_dt": 1.0/30.0,
}
simulation_app = SimulationApp(CONFIG)

# Isaac imports after SimulationApp
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid, VisualSphere, VisualCylinder
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

# Import ONLY from grasp_generator for pose/contacts/orientations
from placement_quality.cube_generalization.grasp_generator import (
    sample_dims,
    six_face_up_orientations,
    sample_contacts_on_faces,
    pose_from_R_t,
)

def generate_contact_metadata(dims, approach_offset=0.01, G_max=0.08):
    """
    Computes up to 18 local contact points on a cuboid's 6 faces (3 per face),
    with metadata for approach, binormal, psi_max, and outward normal.
    """
    metadata = []
    face_axes = {
        '+X': (0,1,2), '-X': (0,1,2),
        '+Y': (1,0,2), '-Y': (1,0,2),
        '+Z': (2,0,1), '-Z': (2,0,1),
    }
    fractions = [0.25, 0.50, 0.75]
    half = dims * 0.5

    for face, (i, j, k) in face_axes.items():
        sign = 1 if face[0] == '+' else -1
        normal   = sign * np.eye(3)[i]      # outward face normal
        approach = -normal                  # gripper approach direction

        ej = np.eye(3)[j]
        ek = np.eye(3)[k]
        du, dv = dims[j], dims[k]
        long_vec, long_len = (ej, du) if du >= dv else (ek, dv)
         # —— NEW: pick the shorter edge as our X‐axis “axis” ——
        axis_vec = ek if du >= dv else ej
        axis     = axis_vec / np.linalg.norm(axis_vec)

        # —— NEW: binormal = approach × axis ——
        binormal = np.cross(axis, approach)
        binormal /= np.linalg.norm(binormal)

       

        base = normal * (half[i] + approach_offset)
        for frac in fractions:
            offset = (frac - 0.5) * long_len
            p_local = base + long_vec * offset
            metadata.append({
                'face':     face,
                'fraction': frac,
                'p_local':  p_local,
                'approach': approach,
                'binormal': binormal,
                'normal':   normal,
                'axis':     axis
            })
    return metadata
# -----------------------------
# Tunables
# -----------------------------
# Single pedestal for the single object
PEDESTAL_CENTER = np.array([0.2, -0.3, 0.05], float)  # cylinder center (x, y, z)
PEDESTAL_RADIUS = 0.08
PEDESTAL_HEIGHT = 0.10

# Lift scene to avoid ground collisions
Z_LIFT = 1.0

# Contacts
CONTACTS_PER_FACE = 3   # we will pick exactly 3 per face
EDGE_MARGIN = 0.005

SPHERE_RADIUS = 0.012
FRAME_SCALE   = 0.06  # unused here, but kept for consistency

FACE_LIST = ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]  # we want 3 points for *each* of these

# -----------------------------

def setup_lighting():
    import omni
    from pxr import UsdLux, UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    def mk_distant(name, intensity, rot_xyz):
        path = f"/World/Lights/{name}"
        light = UsdLux.DistantLight.Define(stage, path)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
    mk_distant("Key",   2200.0, (45.0,   0.0,  0.0))
    mk_distant("Fill1", 1800.0, (-35.0, 35.0, 0.0))
    mk_distant("Fill2", 1500.0, (35.0, -35.0, 0.0))
    mk_distant("SideR", 1200.0, (0.0,   90.0, 0.0))
    mk_distant("SideL", 1200.0, (0.0,  -90.0, 0.0))

def ensure_cuboid(world: World, prim_path: str, size_xyz: np.ndarray, color_rgb=(0.85, 0.85, 0.85)) -> VisualCuboid:
    try:
        obj = world.scene.get_object(prim_path)
        if obj is not None:
            return obj
    except Exception:
        pass
    obj = VisualCuboid(
        prim_path=prim_path,
        name=prim_path.split('/')[-1],
        position=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
        scale=size_xyz,   # keep your style; 4.2 can also use size=
        color=np.array(color_rgb),
    )
    world.scene.add(obj)
    return obj

def ensure_sphere(world: World, prim_path: str, radius: float, color_rgb=(1.0, 0.2, 0.2)) -> VisualSphere:
    try:
        obj = world.scene.get_object(prim_path)
        if obj is not None:
            return obj
    except Exception:
        pass
    obj = VisualSphere(
        prim_path=prim_path,
        name=prim_path.split('/')[-1],
        position=np.zeros(3),
        radius=radius,
        color=np.array(color_rgb),
    )
    world.scene.add(obj)
    return obj

# -----------------------------
# Main
# -----------------------------

def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    setup_lighting()
    set_camera_view(eye=[3.3, 2.8, 2.0 + Z_LIFT], target=[0.2, -0.3, 0.4 + Z_LIFT])

    world.reset()  # clean stage before we place anything

    # Object dims (slightly larger for visibility if you want)
    dims_xyz = np.array(sample_dims(1, seed=77)[0], float) * 1.3

    # Base pose: +Z face up, on top of the (lifted) pedestal
    pedestal_center = PEDESTAL_CENTER.copy()
    pedestal_center[2] += Z_LIFT
    pedestal_top_z = float(pedestal_center[2] + 0.5 * PEDESTAL_HEIGHT)

    # Spawn the single pedestal
    ped = VisualCylinder(
        prim_path="/World/Pedestal",
        name="pedestal",
        position=pedestal_center,
        radius=PEDESTAL_RADIUS,
        height=PEDESTAL_HEIGHT,
        color=np.array([0.6, 0.6, 0.6], float),
    )
    world.scene.add(ped)

    # Base object orientation and pose
    R_map = six_face_up_orientations(spin_degs=(0,))  # spin 0 for clarity
    R_init = R_map["+Z"][0]
    # Put object "sitting" on the pedestal (center z = top + half-height)
    t_base = np.array([PEDESTAL_CENTER[0], PEDESTAL_CENTER[1], pedestal_top_z + dims_xyz[2]/2.0 + 0.0], float)
    pos_base, quat_base = pose_from_R_t(R_init, t_base)

    # Spawn the single object
    cube = ensure_cuboid(world, "/World/Object", dims_xyz, color_rgb=(0.85, 0.85, 0.85))
    cube.set_world_pose(pos_base, quat_base)

    # Make sure dims_xyz is a numpy array of [X, Y, Z] (meters), matching your function.
    meta = generate_contact_metadata(dims_xyz, approach_offset=0.0, G_max=0.08)

    # For each local contact: p_world = R_init @ p_local + t_base
    spheres = []
    for k, m in enumerate(meta):
        p_local = np.asarray(m["p_local"], float).reshape(3)
        p_world = R_init @ p_local + t_base

        sph = ensure_sphere(world, f"/World/contacts/pt_{k:02d}", SPHERE_RADIUS, (1.0, 0.2, 0.2))
        sph.set_world_pose(p_world, np.array([1.0, 0.0, 0.0, 0.0]))
        spheres.append(sph)

    # Make these the defaults so a reset won’t wipe them
    ped.set_default_state(pedestal_center, np.array([1.0, 0.0, 0.0, 0.0]))
    cube.set_default_state(pos_base, quat_base)
    for s in spheres:
        p, _ = s.get_world_pose()
        s.set_default_state(np.array(p), np.array([1.0, 0.0, 0.0, 0.0]))

    # Single reset now that all defaults are set (optional)
    world.reset()
    for _ in range(15):
        world.step(render=False)

    print("18 contact positions visualized (3 per face × 6 faces).")
    try:
        while simulation_app.is_running():
            world.step(render=True)
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
