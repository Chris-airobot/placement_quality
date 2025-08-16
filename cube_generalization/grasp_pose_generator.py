import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence

from scipy.spatial.transform import Rotation as R
from placement_quality.cube_generalization.utils import six_face_up_orientations

INIT_SPIN_DEG = 0
PEDESTAL_HEIGHT = 0.10
PEDESTAL_CENTER_Z = 0.05
PEDESTAL_TOP_Z = PEDESTAL_CENTER_Z + 0.5 * PEDESTAL_HEIGHT # 0.10
WORLD_UP = np.array([0.0, 0.0, 1.0])

# Local surface frames for each cuboid face in the OBJECT frame
_FACE_BASIS = {
    '+X': (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
    '-X': (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), -np.array([1.0, 0.0, 0.0])),
    '+Y': (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    '-Y': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), -np.array([0.0, 1.0, 0.0])),
    '+Z': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    '-Z': (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), -np.array([0.0, 0.0, 1.0])),
}


def face_surface_frame(face: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        u_f, v_f, n_f = _FACE_BASIS[face]
    except KeyError as exc:
        raise ValueError(f"Unknown face label: {face}") from exc
    return u_f.copy(), v_f.copy(), n_f.copy()


def quat_wxyz_to_xyzw(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    assert q.shape[-1] == 4
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    assert q.shape[-1] == 4
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)



def pose_from_R_t(R_wo: np.ndarray, t_wo: np.ndarray) -> Tuple[List[float], List[float]]:
    q_xyzw = R.from_matrix(R_wo).as_quat()
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    return [float(t_wo[0]), float(t_wo[1]), float(t_wo[2])], [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])]


def generate_contact_metadata(
    dims_xyz: np.ndarray,
    approach_offset: float = 0.01,
) -> List[Dict]:
    """
    Generate local contact points and per-face axes for a cuboid.

    - Returns entries with keys: 'face', 'fraction', 'p_local', 'approach', 'binormal', 'normal', 'axis'
    - The closing axis 'axis' is the shorter in-plane edge; 'approach' is -normal; 'binormal' = Z × X.
    """
    metadata: List[Dict] = []
    face_axes = {
        '+X': (0, 1, 2), '-X': (0, 1, 2),
        '+Y': (1, 0, 2), '-Y': (1, 0, 2),
        '+Z': (2, 0, 1), '-Z': (2, 0, 1),
    }
    fractions = [0.25, 0.50, 0.75]
    half = 0.5 * np.asarray(dims_xyz, dtype=float)

    for face, (i, j, k) in face_axes.items():
        sign = 1 if face[0] == '+' else -1
        normal = sign * np.eye(3)[i]
        approach = -normal

        ej = np.eye(3)[j]
        ek = np.eye(3)[k]
        du, dv = float(dims_xyz[j]), float(dims_xyz[k])
        long_vec, long_len = (ej, du) if du >= dv else (ek, dv)
        axis_vec = ek if du >= dv else ej  # shorter edge as closing axis
        axis = axis_vec / np.linalg.norm(axis_vec)

        binormal = np.cross(approach, axis)
        binormal /= (np.linalg.norm(binormal) + 1e-12)

        base = normal * (half[i] + float(approach_offset))
        for frac in fractions:
            offset = (float(frac) - 0.5) * long_len
            p_local = base + long_vec * offset
            metadata.append({
                'face': face,
                'fraction': float(frac),
                'p_local': p_local,
                'approach': approach,
                'binormal': binormal,
                'normal': normal,
                'axis': axis,
            })
    return metadata

def sample_dims(n: int, min_s: float = 0.05, max_s: float = 0.20, seed: int = 0) -> List[Tuple[float, float, float]]:
    """
    Generate exactly n box dimension triplets (x, y, z) in metres with simple shape bias
    while enforcing constraints:
      - All edges lie within [min_s, max_s]
      - Each triplet has at least one edge exactly equal to min_s
    """
    rng = np.random.default_rng(seed)

    min_s = float(min_s)
    max_s = float(max_s)
    assert max_s >= min_s, "max_s must be >= min_s"

    def enforce_constraints(dims: np.ndarray) -> Tuple[float, float, float]:
        # Clip to bounds
        dims = np.clip(dims.astype(float), min_s, max_s)
        # Ensure at least one dimension equals min_s
        if not np.any(np.isclose(dims, min_s)):
            idx = int(np.argmin(dims))
            dims[idx] = min_s
        return float(dims[0]), float(dims[1]), float(dims[2])

    # Shape types to encourage some variety without breaking bounds
    shape_types = ("thin", "long", "tall", "cubey", "random")
    results: List[Tuple[float, float, float]] = []

    # Helper to sample a biased triplet within range
    span = max_s - min_s

    def sample_one(kind: str) -> Tuple[float, float, float]:
        if kind == "thin":
            # Two arbitrary, one near the lower end
            a = rng.uniform(min_s, max_s)
            b = rng.uniform(min_s, max_s)
            c = rng.uniform(min_s, min_s + 0.3 * span)
            dims = np.array([a, b, c], dtype=float)
        elif kind == "long":
            # One near upper end, others arbitrary
            hi = rng.uniform(min_s + 0.7 * span, max_s)
            lo1 = rng.uniform(min_s, max_s)
            lo2 = rng.uniform(min_s, max_s)
            # Randomize which axis is the long one
            idx = rng.integers(0, 3)
            vals = [lo1, lo2, rng.uniform(min_s, max_s)]
            vals[idx] = hi
            dims = np.array(vals, dtype=float)
        elif kind == "tall":
            # Similar to long: emphasize z
            a = rng.uniform(min_s, max_s)
            b = rng.uniform(min_s, max_s)
            c = rng.uniform(min_s + 0.7 * span, max_s)
            dims = np.array([a, b, c], dtype=float)
        elif kind == "cubey":
            s = rng.uniform(min_s, max_s)
            dims = np.array([s * rng.uniform(0.9, 1.1),
                             s * rng.uniform(0.9, 1.1),
                             s * rng.uniform(0.9, 1.1)], dtype=float)
        else:  # random
            dims = rng.uniform(min_s, max_s, size=3).astype(float)
        return enforce_constraints(dims)

    for i in range(n):
        kind = shape_types[int(rng.integers(0, len(shape_types)))]
        results.append(sample_one(kind))

    return results


def build_tool_orientation_from_meta(
    meta: Dict,
    R_obj_to_world: np.ndarray,
) -> np.ndarray:
    """
    Build a right-handed tool frame from contact metadata.

    Returns a 3x3 rotation matrix whose columns are [X_tool, Y_tool, Z_tool] in WORLD.
    """
    face = meta['face']
    n_f_obj = face_surface_frame(face)[2]
    z_loc = -n_f_obj

    # Start with the stored axis (shorter face edge)
    x_raw = np.asarray(meta['axis'], dtype=float)
    x_loc = x_raw - np.dot(x_raw, z_loc) * z_loc
    norm_x = np.linalg.norm(x_loc)
    if norm_x < 1e-12:
        u_f, v_f, _ = face_surface_frame(face)
        x_loc = u_f if np.linalg.norm(u_f) > np.linalg.norm(v_f) else v_f
    x_loc = x_loc / np.linalg.norm(x_loc)

    # Make Y = stored axis projected into the plane; complete the frame
    y_raw = np.asarray(meta['axis'], dtype=float)
    y_loc = y_raw - np.dot(y_raw, z_loc) * z_loc
    norm_y = np.linalg.norm(y_loc)
    if norm_y < 1e-12:
        u_f, v_f, _ = face_surface_frame(face)
        y_loc = u_f if np.linalg.norm(u_f) < np.linalg.norm(v_f) else v_f
    y_loc = y_loc / np.linalg.norm(y_loc)

    x_loc = np.cross(y_loc, z_loc); x_loc /= np.linalg.norm(x_loc)
    y_loc = np.cross(z_loc, x_loc); y_loc /= np.linalg.norm(y_loc)

    # Align X with stored meta axis direction
    x_meta = np.asarray(meta['axis'], dtype=float)
    if float(np.dot(x_loc, x_meta)) < 0.0:
        x_loc *= -1.0
        y_loc *= -1.0

    x_w = R_obj_to_world @ x_loc
    y_w = R_obj_to_world @ y_loc
    z_w = R_obj_to_world @ z_loc

    R_tool = np.column_stack([x_w, y_w, z_w])
    if float(np.dot(np.cross(x_w, y_w), z_w)) < 0.0:
        y_w *= -1.0
        R_tool = np.column_stack([x_w, y_w, z_w])
    return R_tool


def _quat_wxyz_from_R(Rm: np.ndarray) -> np.ndarray:
    q_xyzw = R.from_matrix(Rm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def _apply_local_rotations(
    R_tool: np.ndarray,
    enable_tilt: bool,
    tilt_deg: float,
    enable_yaw: bool,
    yaw_deg: float,
    enable_roll: bool,
    roll_deg: float,
) -> np.ndarray:
    R_out = R_tool
    if enable_tilt:
        tilt_rad = float(np.deg2rad(tilt_deg - 90.0))
        if abs(tilt_rad) > 1e-9:
            R_tilt = R.from_rotvec(tilt_rad * R_out[:, 0]).as_matrix()
            R_out = R_tilt @ R_out
    if enable_roll:
        roll_rad = float(np.deg2rad(roll_deg))
        if abs(roll_rad) > 1e-9:
            R_roll = R.from_rotvec(roll_rad * R_out[:, 1]).as_matrix()
            R_out = R_roll @ R_out
    if enable_yaw:
        yaw_rad = float(np.deg2rad(yaw_deg))
        if abs(yaw_rad) > 1e-9:
            R_yaw = R.from_rotvec(yaw_rad * R_out[:, 2]).as_matrix()
            R_out = R_yaw @ R_out
    return R_out


def generate_grasp_poses(
    dims_xyz: np.ndarray,
    R_obj_to_world: np.ndarray,
    t_obj_world: np.ndarray,
    enable_tilt: bool = False,
    tilt_deg: float = 90.0,
    enable_yaw: bool = False,
    yaw_deg: float = 0.0,
    enable_roll: bool = False,
    roll_deg: float = 0.0,
    filter_by_gripper_open: bool = True,
    gripper_open_max: float = 0.08,
    apply_hand_to_tcp: bool = False,
    hand_to_tcp_z: float = 0.1034,
    extra_insert: float = 0.0,
) -> List[Dict]:
    """
    Generate grasp pose candidates (position + orientation) for a cuboid object.

    Inputs:
      - dims_xyz: object dimensions [dx, dy, dz]
      - R_obj_to_world: 3x3 rotation of the object in world
      - t_obj_world: 3-vector translation of the object in world
      - Bottom face is excluded automatically (based on WORLD_UP)
      - enable_tilt/yaw/roll + angles: apply individually or in combination
      - apply_hand_to_tcp: if True, also return panda_hand pose with an orientation-aware
        offset along local -Z scaled by 1/cos(theta)

    Returns a list of dicts containing:
      - 'face', 'fraction', 'contact_position_world'
      - 'tool_quaternion_wxyz', 'tool_rotation', 'face_normal_world'
      - Optionally 'hand_position_world', 'hand_quaternion_wxyz' when apply_hand_to_tcp is True
    """
    dims_xyz = np.asarray(dims_xyz, dtype=float)
    R_obj_to_world = np.asarray(R_obj_to_world, dtype=float)
    t_obj_world = np.asarray(t_obj_world, dtype=float)

    contacts = generate_contact_metadata(dims_xyz, approach_offset=0.01)

    # Auto-detect and exclude the bottom face (most opposite to WORLD_UP)
    face_to_normal_obj = {
        '+X': np.array([ 1.0, 0.0, 0.0]),
        '-X': np.array([-1.0, 0.0, 0.0]),
        '+Y': np.array([ 0.0, 1.0, 0.0]),
        '-Y': np.array([ 0.0,-1.0, 0.0]),
        '+Z': np.array([ 0.0, 0.0, 1.0]),
        '-Z': np.array([ 0.0, 0.0,-1.0]),
    }
    dot_to_face: Dict[float, str] = {}
    for face, n_obj in face_to_normal_obj.items():
        n_world = R_obj_to_world @ n_obj
        dot_to_face[float(np.dot(n_world, WORLD_UP))] = face
    # Small numerical drift safety: pick min dot value (most downward)
    min_dot = min(dot_to_face.keys())
    bottom_face = dot_to_face[min_dot]
    contacts = [m for m in contacts if m['face'] != bottom_face]

    results: List[Dict] = []
    for meta in contacts:
        # Filter by gripper max opening using the stored closing axis (object frame)
        if filter_by_gripper_open:
            half_extents = 0.5 * dims_xyz
            axis_obj = np.asarray(meta['axis'], dtype=float)
            jaw_span = float(2.0 * np.sum(np.abs(axis_obj) * half_extents))
            if jaw_span > float(gripper_open_max):
                continue

        # Contact position in world
        p_local = np.asarray(meta['p_local'], dtype=float)
        p_world = R_obj_to_world @ p_local + t_obj_world

        # Base orientation from surface frame, then apply local rotations
        R_tool = build_tool_orientation_from_meta(meta, R_obj_to_world)
        R_tool = _apply_local_rotations(
            R_tool,
            enable_tilt, tilt_deg,
            enable_yaw, yaw_deg,
            enable_roll, roll_deg,
        )
        q_wxyz = _quat_wxyz_from_R(R_tool)

        # Face normal in world (outward)
        n_face_world = R_obj_to_world @ face_surface_frame(meta['face'])[2]

        result: Dict = {
            'face': meta['face'],
            'fraction': float(meta['fraction']),
            'contact_position_world': p_world.astype(float),
            'tool_quaternion_wxyz': q_wxyz.astype(float),
            'tool_rotation': R_tool.astype(float),
            'face_normal_world': n_face_world.astype(float),
        }

        if apply_hand_to_tcp:
            # Orientation-aware offset to convert TCP to panda_hand
            z_tool_w = R_tool[:, 2]
            cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_world)))
            z_local = -(float(hand_to_tcp_z) + float(extra_insert)) / cos_theta
            hand_position_world = p_world + R_tool @ np.array([0.0, 0.0, z_local], dtype=float)
            result['hand_position_world'] = hand_position_world.astype(float)
            result['hand_quaternion_wxyz'] = q_wxyz.astype(float)

        results.append(result)

    return results



def generate_grasp_poses_including_bottom(
    dims_xyz: np.ndarray,
    R_obj_to_world: np.ndarray,
    t_obj_world: np.ndarray,
    enable_tilt: bool = False,
    tilt_deg: float = 90.0,
    enable_yaw: bool = False,
    yaw_deg: float = 0.0,
    enable_roll: bool = False,
    roll_deg: float = 0.0,
    filter_by_gripper_open: bool = False,   # do NOT gate for training
    gripper_open_max: float = 0.08,
    apply_hand_to_tcp: bool = True,
    hand_to_tcp_z: float = 0.1034,
    extra_insert: float = -0.0334,
) -> List[Dict]:
    dims_xyz = np.asarray(dims_xyz, dtype=float)
    R_obj_to_world = np.asarray(R_obj_to_world, dtype=float)
    t_obj_world = np.asarray(t_obj_world, dtype=float)

    # Keep ALL contacts, including bottom face
    contacts = generate_contact_metadata(dims_xyz, approach_offset=0.01)

    results: List[Dict] = []
    for meta in contacts:
        if filter_by_gripper_open:
            half_extents = 0.5 * dims_xyz
            axis_obj = np.asarray(meta['axis'], dtype=float)
            jaw_span = float(2.0 * np.sum(np.abs(axis_obj) * half_extents))
            if jaw_span > float(gripper_open_max):
                continue

        p_local = np.asarray(meta['p_local'], dtype=float)
        p_world = R_obj_to_world @ p_local + t_obj_world

        R_tool = build_tool_orientation_from_meta(meta, R_obj_to_world)
        R_tool = _apply_local_rotations(
            R_tool,
            enable_tilt, tilt_deg,
            enable_yaw, yaw_deg,
            enable_roll, roll_deg,
        )
        q_wxyz = _quat_wxyz_from_R(R_tool)

        n_face_world = R_obj_to_world @ face_surface_frame(meta['face'])[2]
        result: Dict = {
            'face': meta['face'],
            'fraction': float(meta['fraction']),
            'contact_position_world': p_world.astype(float),
            'tool_quaternion_wxyz': q_wxyz.astype(float),
            'tool_rotation': R_tool.astype(float),
            'face_normal_world': n_face_world.astype(float),
        }

        if apply_hand_to_tcp:
            z_tool_w = R_tool[:, 2]
            cos_theta = max(1e-3, -float(np.dot(z_tool_w, n_face_world)))
            z_local = -(float(hand_to_tcp_z) + float(extra_insert)) / cos_theta
            hand_position_world = p_world + R_tool @ np.array([0.0, 0.0, z_local], dtype=float)
            result['hand_position_world'] = hand_position_world.astype(float)
            result['hand_quaternion_wxyz'] = q_wxyz.astype(float)
            result['axis_obj'] = np.asarray(meta['axis'], dtype=float)  # NEW
            result['p_local']  = p_local.astype(float) 

        results.append(result)
    return results








if __name__ == "__main__":
    dims_xyz = np.array([0.111, 0.149, 0.05], dtype=float)
    R_init = six_face_up_orientations(spin_degs=(INIT_SPIN_DEG,))["+Z"][0]
    t_obj = np.array([0.0, 0.0, PEDESTAL_TOP_Z], dtype=float)
    ENABLE_TILT = True
    TILT_DEG = 75.0

    # Yaw controls (about local Z/approach axis)
    ENABLE_YAW = True
    YAW_DEG = 15.0

    # Roll controls (about local Y/binormal axis)
    ENABLE_ROLL = True
    ROLL_DEG = 15.0

    # Hand-to-TCP offset and optional extra insert along local -Z
    HAND_TO_TCP_Z = 0.1034
    EXTRA_INSERT = -0.0334

    grasps = generate_grasp_poses(
        dims_xyz=dims_xyz,
        R_obj_to_world=R_init,
        t_obj_world=t_obj,
        enable_tilt=True,  tilt_deg=TILT_DEG,
        enable_yaw=True,   yaw_deg=YAW_DEG,
        enable_roll=True,  roll_deg=ROLL_DEG,
        apply_hand_to_tcp=True,
        hand_to_tcp_z=HAND_TO_TCP_Z,
        extra_insert=EXTRA_INSERT,  # match your current setting
    )

    for g in grasps:
        p_world = g['contact_position_world']
        q_tool_wxyz = g['tool_quaternion_wxyz']
        # Or if you want panda_hand poses directly:
        hand_pos = g['hand_position_world']
        hand_quat = g['hand_quaternion_wxyz']