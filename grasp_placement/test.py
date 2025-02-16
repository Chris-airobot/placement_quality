import numpy as np
import math
import tf_transformations as tft

# These are the "base" orientations in Euler angles [roll, pitch, yaw] (degrees) you said:
LOCAL_FACE_AXES = {
    "+z":  np.array([  0.0,   0.0,   0.0]),     # top face up  z diff
    "-z":  np.array([-180.0, 0.0,   0.0]),     # bottom face up z diff
    "+y":  np.array([ 90.0,  0.0,   0.0]),     # left face up y diff
    "-y":  np.array([-90.0,  0.0,   0.0]),     # right face up y diff
    "+x":  np.array([ 90.0,  0.0,  90.0]),     # front face up y diff
    "-x":  np.array([ 90.0,  0.0, -90.0]),     # back face up y diff
}

def compute_feasible_cube_pose(data, cube_size=0.05):
    """
    1) Compute the predicted final cube orientation from:
         T_pred = T_ee_target * (inv(T_ee_current) * T_cube_current)
    2) Convert that orientation to Euler angles (degrees).
    3) Among the 6 base orientations in LOCAL_FACE_AXES, pick whichever is
       closest in quaternion distance to q_pred.
    4) Compute the orientation difference q_diff = q_pred * inv(q_base).
       Convert to Euler, keep only one axis of that difference (e.g. pitch),
       zero out the other two. Then recompose the difference.
    5) Final orientation q_final = q_diff_modified * q_base.
    6) Overwrite (x,y) with data["cube_target_position"], set z = cube_size/2,
       then return as {"position": [...], "orientation": [w,x,y,z]}.
    """
    # ------------------------------------------------------------
    # (A) Extract relevant data
    # ------------------------------------------------------------
    ee_pos_current = data["ee_position"]            # [x,y,z]
    ee_quat_current_wxyz = data["ee_orientation"]   # [w,x,y,z]

    cube_pos_current = data["cube_position"]        # [x,y,z]
    cube_quat_current_wxyz = data["cube_orientation"]   # [w,x,y,z]

    # Suppose the target EE orientation is given as Euler angles:
    ee_target_euler = data["ee_target_orientation"]   # [roll, pitch, yaw] in radians
    q_ee_target_xyzw = tft.quaternion_from_euler(
        ee_target_euler[0],
        ee_target_euler[1],
        ee_target_euler[2],
        axes='sxyz'
    )  # [x,y,z,w]

    # The final desired x,y position of the cube
    cube_target_pos = data["cube_target_position"]   # [x, y, z], ignoring z

    # ------------------------------------------------------------
    # (B) Build transforms
    # ------------------------------------------------------------
    q_ee_current_xyzw   = convert_wxyz_to_xyzw(ee_quat_current_wxyz)
    q_cube_current_xyzw = convert_wxyz_to_xyzw(cube_quat_current_wxyz)

    T_ee_current   = build_transform(ee_pos_current,   q_ee_current_xyzw)
    T_cube_current = build_transform(cube_pos_current, q_cube_current_xyzw)

    # Relative transform:  T_relative = inv(T_ee_current) * T_cube_current
    T_relative = np.linalg.inv(T_ee_current) @ T_cube_current

    # Build T_ee_target: ignoring any target EE position, only orientation
    T_ee_target = build_transform([0,0,0], q_ee_target_xyzw)

    # Predicted cube transform in world
    T_pred = T_ee_target @ T_relative

    # ------------------------------------------------------------
    # (C) Convert T_pred to a quaternion and Euler angles (degrees)
    # ------------------------------------------------------------
    q_pred_xyzw = tft.quaternion_from_matrix(T_pred)   # [x,y,z,w]
    r_pred, p_pred, y_pred = tft.euler_from_quaternion(q_pred_xyzw, axes='sxyz')
    r_pred_deg = math.degrees(r_pred)
    p_pred_deg = math.degrees(p_pred)
    y_pred_deg = math.degrees(y_pred)

    # ------------------------------------------------------------
    # (D) Find the "base orientation" among LOCAL_FACE_AXES that is
    #     closest to q_pred in quaternion space
    # ------------------------------------------------------------
    best_key = None
    best_dist = 1e9
    best_q_base_xyzw = None

    for key, euler_deg in LOCAL_FACE_AXES.items():
        # 1) Convert the euler_deg -> radians -> quaternion base
        r_base_rad = math.radians(euler_deg[0])
        p_base_rad = math.radians(euler_deg[1])
        y_base_rad = math.radians(euler_deg[2])
        q_base_xyzw = tft.quaternion_from_euler(r_base_rad, p_base_rad, y_base_rad, axes='sxyz')
        
        # 2) measure distance
        dist = quaternion_distance(q_pred_xyzw, q_base_xyzw)
        if dist < best_dist:
            best_dist = dist
            best_key = key
            best_q_base_xyzw = q_base_xyzw

    # ------------------------------------------------------------
    # (E) Compute orientation difference: q_diff = q_pred * inv(q_base)
    #     Then convert to Euler, keep only one axis (e.g. pitch), zero out the others
    # ------------------------------------------------------------
    # By definition, if q_pred = q_diff * q_base, then q_diff = q_pred * q_base^-1
    q_base_inv = tft.quaternion_inverse(best_q_base_xyzw)
    q_diff_xyzw = tft.quaternion_multiply(q_pred_xyzw, q_base_inv)

    # Convert that difference to Euler angles
    r_diff, p_diff, y_diff = tft.euler_from_quaternion(q_diff_xyzw, axes='sxyz')
    r_diff_deg = math.degrees(r_diff)
    p_diff_deg = math.degrees(p_diff)
    y_diff_deg = math.degrees(y_diff)

    # Suppose we only preserve pitch difference, zero out roll & yaw
    # (You can choose whichever axis you want to keep or partially keep)
    if best_key in ["+z", "-z"]:
        # Preserve yaw -> zero out roll & pitch
        r_diff_deg_mod = 0.0
        p_diff_deg_mod = 0.0
        y_diff_deg_mod = y_diff_deg
    else:
        # Preserve pitch -> zero out roll & yaw
        r_diff_deg_mod = 0.0
        p_diff_deg_mod = p_diff_deg
        y_diff_deg_mod = 0.0

    # Recompose q_diff_modified
    q_diff_modified_xyzw = tft.quaternion_from_euler(
        math.radians(r_diff_deg_mod),
        math.radians(p_diff_deg_mod),
        math.radians(y_diff_deg_mod),
        axes='sxyz'
    )

    # Final orientation q_final = q_diff_modified * q_base
    q_final_xyzw = tft.quaternion_multiply(q_diff_modified_xyzw, best_q_base_xyzw)

    # ------------------------------------------------------------
    # (F) Build final transform, override x,y, set z => ground
    # ------------------------------------------------------------
    T_final = tft.quaternion_matrix(q_final_xyzw)
    # Use T_pred translation as a base
    T_final[:3, 3] = T_pred[:3, 3].copy()

    # Overwrite x,y with target, place the cube on the ground
    T_final[0, 3] = cube_target_pos[0]
    T_final[1, 3] = cube_target_pos[1]
    T_final[2, 3] = cube_size / 2.0

    # ------------------------------------------------------------
    # (G) Convert to [x,y,z] + quaternion [w,x,y,z]
    # ------------------------------------------------------------
    final_pos, final_quat_wxyz = transform_to_pose(T_final)

    return final_quat_wxyz


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def build_transform(pos_xyz, quat_xyzw):
    """
    Build 4x4 from position [x,y,z] and quaternion [x,y,z,w].
    """
    T = tft.quaternion_matrix(quat_xyzw)
    T[:3, 3] = pos_xyz
    return T

def transform_to_pose(T):
    """
    Convert 4x4 -> (position, orientation [w,x,y,z])
    """
    pos = T[:3, 3].tolist()
    q_xyzw = tft.quaternion_from_matrix(T)  # [x,y,z,w]
    return pos, [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

def convert_wxyz_to_xyzw(q_wxyz):
    """[w,x,y,z] -> [x,y,z,w]."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

def quaternion_distance(q1_xyzw, q2_xyzw):
    """
    Simple measure: 1 - |dot(q1,q2)| for unit quaternions q1,q2 in [x,y,z,w] form.
    Ranges [0..2].
    """
    dot_val = abs(np.dot(q1_xyzw, q2_xyzw))
    return 1.0 - dot_val
