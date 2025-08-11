import json
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from placement_quality.cube_generalization.grasp_pose_generator import (
    sample_dims,
    six_face_up_orientations,
    pose_from_R_t,
    generate_grasp_poses,
)


# Environment and clearance parameters (align with visualization)
PEDESTAL_HEIGHT = 0.10
PEDESTAL_CENTER_Z = 0.05
PEDESTAL_TOP_Z = PEDESTAL_CENTER_Z + 0.5 * PEDESTAL_HEIGHT

MIN_CLEARANCE = 0.02
PALM_DEPTH = 0.038
FINGER_THICK = 0.03
GRIPPER_OPEN_MAX = 0.08
MIN_PALM_CLEAR = 0.01
APPROACH_BACKOFF = 0.09

# Grasp transform parameters
ENABLE_TILT = True
TILT_DEG = 75.0
ENABLE_YAW = True
YAW_DEG = 15.0
ENABLE_ROLL = True
ROLL_DEG = 15.0

# Panda hand offset (hand->TCP along local +Z) and extra insert along -Z
HAND_TO_TCP_Z = 0.1034
EXTRA_INSERT = -0.0334

# Initial and final orientation sampling
INIT_SPINS_DEG: Tuple[int, ...] = (0, 90)
FINAL_SPIN_CANDIDATES_DEG: Tuple[int, ...] = (0, 90, 180, 270)
N_FINAL_SPINS_PER_FACE: int = 1
FACE_LIST: Tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def passes_clearance(
    p_world: np.ndarray,
    ped_top_z: float,
    R_tool: np.ndarray,
    min_contact_clear: float = MIN_CLEARANCE,
    min_palm_clear: float = MIN_PALM_CLEAR,
    palm_depth: float = PALM_DEPTH,
    approach_backoff: float = APPROACH_BACKOFF,
    finger_thick: float = FINGER_THICK,
) -> bool:
    """Match visualization clearance test (contact and palm-bottom clearance)."""
    if (float(p_world[2]) - float(ped_top_z)) < float(min_contact_clear):
        return False
    z_w = R_tool[:, 2]
    y_w = R_tool[:, 1]
    effective_palm_depth = max(float(palm_depth) - float(approach_backoff), 0.0)
    palm_center_z = float(p_world[2]) - effective_palm_depth * float(z_w[2])
    palm_bottom_z = palm_center_z - float(finger_thick) * abs(float(y_w[2]))
    return (palm_bottom_z - float(ped_top_z)) >= float(min_palm_clear)


def generate_experiments(
    num_objects: int = 4,
    dims_min: float = 0.05,
    dims_max: float = 0.20,
    seed: int = 980579,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    test_data: List[Dict] = []

    # Sample diverse object dimensions
    box_dim_lists = sample_dims(n=num_objects, min_s=dims_min, max_s=dims_max, seed=seed)
    print(box_dim_lists)

    for obj_idx, dims in enumerate(box_dim_lists):
        dims_xyz = np.array(dims, dtype=float)

        # Initial orientation candidates: multiple faces and spins
        R_init_map = six_face_up_orientations(spin_degs=INIT_SPINS_DEG)
        for init_face, R_list in R_init_map.items():
            for R_init in R_list:
                R_init = np.asarray(R_init, dtype=float)

                # Place object at pedestal with correct Z (bottom on pedestal top)
                half_extents = 0.5 * dims_xyz
                h_z_init = float(np.sum(np.abs(R_init[2, :]) * half_extents))
                t_init = np.array([0.2, -0.3, PEDESTAL_TOP_Z + h_z_init], dtype=float)
                pos_i, quat_i = pose_from_R_t(R_init, t_init)

                # Build grasp candidates for all 8 transform variants
                transform_variants = [
                    {"tilt": False, "yaw": False, "roll": False},
                    {"tilt": True,  "yaw": False, "roll": False},
                    {"tilt": False, "yaw": True,  "roll": False},
                    {"tilt": False, "yaw": False, "roll": True },
                    # {"tilt": True,  "yaw": True,  "roll": False},
                    # {"tilt": True,  "yaw": False, "roll": True },
                    # {"tilt": False, "yaw": True,  "roll": True },
                    {"tilt": True,  "yaw": True,  "roll": True },
                ]

                grasps_by_variant: List[Tuple[Dict, List[Dict]]] = []
                for tv in transform_variants:
                    grasps_tv = generate_grasp_poses(
                        dims_xyz=dims_xyz,
                        R_obj_to_world=R_init,
                        t_obj_world=t_init,
                        enable_tilt=tv["tilt"], tilt_deg=TILT_DEG,
                        enable_yaw=tv["yaw"],  yaw_deg=YAW_DEG,
                        enable_roll=tv["roll"], roll_deg=ROLL_DEG,
                        filter_by_gripper_open=True, gripper_open_max=GRIPPER_OPEN_MAX,
                        apply_hand_to_tcp=True,
                        hand_to_tcp_z=HAND_TO_TCP_Z,
                        extra_insert=EXTRA_INSERT,
                    )

                    valid_grasps_tv: List[Dict] = []
                    for g in grasps_tv:
                        p_world = np.asarray(g['contact_position_world'], dtype=float)
                        R_tool = np.asarray(g['tool_rotation'], dtype=float)
                        if passes_clearance(p_world, PEDESTAL_TOP_Z, R_tool):
                            valid_grasps_tv.append(g)
                    # Keep empty lists too; we still want coverage info per variant
                    grasps_by_variant.append((tv, valid_grasps_tv))

                # Final orientation candidates: select N spins per face
                spins_this_block = tuple(sorted(rng.choice(FINAL_SPIN_CANDIDATES_DEG, size=N_FINAL_SPINS_PER_FACE, replace=False)))
                R_final_map = six_face_up_orientations(spin_degs=spins_this_block)

                for fin_face in FACE_LIST:
                    for R_fin in R_final_map[fin_face]:
                        R_fin = np.asarray(R_fin, dtype=float)
                        # Place final object bottom on pedestal top as well
                        h_z_fin = float(np.sum(np.abs(R_fin[2, :]) * half_extents))
                        t_fin = np.array([0.2, -0.3, PEDESTAL_TOP_Z + h_z_fin], dtype=float)
                        pos_f, quat_f = pose_from_R_t(R_fin, t_fin)

                        # One block of trials per transform variant, sharing the same init/final
                        for tv, valid_grasps_tv in grasps_by_variant:
                            for g in valid_grasps_tv:
                                trial = {
                                    "object_dimensions": list(map(float, dims_xyz.tolist())),
                                    "initial_object_pose": list(map(float, pos_i + quat_i)),
                                    "final_object_pose": list(map(float, pos_f + quat_f)),
                                    # Save panda_hand pose for robot execution
                                    "grasp_pose": list(map(float, (g['hand_position_world'].tolist() + g['hand_quaternion_wxyz'].tolist()))),
                                    # Debug: which transform toggles generated this grasp
                                    "debug_info": {
                                        "face": g['face'],
                                        "fraction": float(g['fraction']),
                                        "face_normal_world": np.asarray(g['face_normal_world'], dtype=float).tolist(),
                                        "tilt_enabled": bool(tv["tilt"]),
                                        "yaw_enabled": bool(tv["yaw"]),
                                        "roll_enabled": bool(tv["roll"]),
                                        "tilt_deg": float(TILT_DEG),
                                        "yaw_deg": float(YAW_DEG),
                                        "roll_deg": float(ROLL_DEG),
                                    },
                                }
                                test_data.append(trial)

    return test_data


def main():
    data = generate_experiments(
        num_objects=4,
        dims_min=0.05,
        dims_max=0.20,
        seed=980579,
    )

    out_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiments.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} trials → {out_path}")


if __name__ == "__main__":
    main()

