
from learning_models.process_data_helpers import *

tf = [{"parent_frame": "world", "child_frame": "panda_link0", "translation": {"x": -2.38418573772492e-09, "y": 2.98023217215615e-10, "z": 2.384185648907078e-08}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, {"parent_frame": "panda_link0", "child_frame": "panda_link1", "translation": {"x": 0.0, "y": -1.7763568394002505e-15, "z": 0.33299991488456726}, "rotation": {"x": 6.409251085415235e-08, "y": 0.0, "z": 0.537647008895874, "w": 0.8431699872016907}}, {"parent_frame": "panda_link1", "child_frame": "panda_link2", "translation": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"x": -0.6772404313087463, "y": 0.20333564281463623, "z": 0.20333567261695862, "w": 0.6772404909133911}}, {"parent_frame": "panda_link2", "child_frame": "panda_link3", "translation": {"x": 2.2351741790771484e-08, "y": -0.3160000443458557, "z": -3.725290298461914e-08}, "rotation": {"x": 0.6846010684967041, "y": -0.17697840929031372, "z": 0.17697837948799133, "w": 0.6846010684967041}}, {"parent_frame": "panda_link3", "child_frame": "panda_link4", "translation": {"x": 0.08249995112419128, "y": 1.4901161193847656e-08, "z": 8.381903171539307e-09}, "rotation": {"x": 0.1680595576763153, "y": 0.6868448853492737, "z": -0.6868449449539185, "w": 0.16805973649024963}}, {"parent_frame": "panda_link4", "child_frame": "panda_link5", "translation": {"x": -0.08249993622303009, "y": 0.38399988412857056, "z": -2.9802322387695312e-08}, "rotation": {"x": -0.4988028109073639, "y": -0.501194417476654, "z": -0.501194417476654, "w": 0.4988027811050415}}, {"parent_frame": "panda_link5", "child_frame": "panda_link6", "translation": {"x": 0.0, "y": 0.0, "z": -0.0}, "rotation": {"x": 0.48890674114227295, "y": -0.510852575302124, "z": 0.5108525156974792, "w": 0.4889066517353058}}, {"parent_frame": "panda_link6", "child_frame": "panda_link7", "translation": {"x": 0.08799996227025986, "y": 3.725290298461914e-09, "z": -3.3527612686157227e-08}, "rotation": {"x": 0.6473238468170166, "y": -0.2845557928085327, "z": 0.2845558524131775, "w": 0.6473236680030823}}, {"parent_frame": "panda_link7", "child_frame": "panda_link8", "translation": {"x": -4.6566128730773926e-09, "y": -3.725290298461914e-09, "z": 0.10699998587369919}, "rotation": {"x": 0.0, "y": 7.450580596923828e-09, "z": 0.0, "w": 1.0000001192092896}}, {"parent_frame": "panda_link8", "child_frame": "panda_hand", "translation": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 2.7430360205471516e-09, "z": -0.3826834559440613, "w": 0.9238795042037964}}, {"parent_frame": "panda_hand", "child_frame": "panda_leftfinger", "translation": {"x": -4.172761691734195e-09, "y": 0.03234317526221275, "z": 0.05839996039867401}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.9999999403953552}}, {"parent_frame": "panda_hand", "child_frame": "panda_rightfinger", "translation": {"x": 7.025846571195871e-09, "y": -0.032340649515390396, "z": 0.05839996412396431}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.9999999403953552}}, {"parent_frame": "world", "child_frame": "Cube", "translation": {"x": 0.1289782077074051, "y": 0.2918505370616913, "z": 0.02574998140335083}, "rotation": {"x": -7.305934701662409e-08, "y": 5.024838500844453e-08, "z": -2.6920994855572644e-07, "w": 1.0}}]

import numpy as np
import tf_transformations as tft


def get_transform_to_world(tf_list, target_frame):
    """
    Recursively computes the transform from the 'world' frame to target_frame.
    
    Each TF entry in tf_list is assumed to have:
      - "parent_frame": The parent frame name.
      - "child_frame": The child frame name.
      - "translation": A dict with keys "x", "y", "z".
      - "rotation": A dict with keys "x", "y", "z", "w" (in wxyz order in the data).
    
    Returns a 4x4 homogeneous transformation matrix representing the pose of 
    target_frame in the world coordinate system, or None if no chain can be built.
    """
    if target_frame == "world":
        return np.eye(4)
    
    # Look for an entry whose child_frame is the target_frame.
    for tf_entry in tf_list:
        if tf_entry.get("child_frame") == target_frame:
            # Get the transform from its parent to this target_frame.
            translation = tf_entry["translation"]
            rotation = tf_entry["rotation"]
            pos = np.array([translation["x"], translation["y"], translation["z"]])
            # Convert quaternion from the tf data order (wxyz) to (x,y,z,w) order.
            quat_wxyz = np.array([rotation["w"], rotation["x"], rotation["y"], rotation["z"]])
            T_target_given_parent = pose_to_homogeneous(pos, quat_wxyz)
            
            parent_frame = tf_entry["parent_frame"]
            # Recursively get the transform from world to the parent_frame.
            T_parent = get_transform_to_world(tf_list, parent_frame)
            if T_parent is None:
                # Could not resolve parent transform.
                return None
            # The complete transform is the chain: T_world->target = T_world->parent * T_parent->target.
            return np.dot(T_parent, T_target_given_parent)
    # If no entry is found for the target_frame, return None.
    return None

def get_relative_transform(tf_list, source_frame, target_frame):
    """
    Computes the relative transform from source_frame to target_frame.
    
    The transformation is computed as:
       T_target_in_source = (T_source_in_world)^{-1} * T_target_in_world
       
    Returns the 4x4 homogeneous transformation matrix representing the pose 
    of target_frame in the coordinate system of source_frame.
    """
    T_source = get_transform_to_world(tf_list, source_frame)
    T_target = get_transform_to_world(tf_list, target_frame)
    
    if T_source is None:
        print(f"Transform chain to source frame '{source_frame}' could not be resolved.")
        return None
    if T_target is None:
        print(f"Transform chain to target frame '{target_frame}' could not be resolved.")
        return None
    
    # Compute the relative transform.
    T_relative = np.dot(np.linalg.inv(T_source), T_target)
    return T_relative

# --- Example Usage ---


# For example, to get the transform from world to 'panda_hand':
T_panda_hand = get_transform_to_world(tf, "panda_hand")
print("Transform from world to panda_hand:")
print(T_panda_hand)

# And to compute the relative transform between, say, 'panda_hand' (source) and 'Cube' (target):
T_relative = get_relative_transform(tf, "panda_link8", "panda_hand")
if T_relative is not None:
    print("\nRelative transform from panda_hand to Cube:")
    print(T_relative)




# Your given 4x4 homogeneous transformation matrix.
T = np.array([
    [ 7.07106735e-01,  7.07106827e-01,  5.06846976e-09,  0.00000000e+00],
    [-7.07106827e-01,  7.07106735e-01, -2.09942910e-09,  0.00000000e+00],
    [-5.06846981e-09, -2.09942903e-09,  1.00000000e+00,  3.46944695e-18],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])

# Extract the translation vector (x, y, z)
translation = T[0:3, 3]

# Extract the quaternion from the rotation part.
# Note: tf.transformations.quaternion_from_matrix returns [x, y, z, w]
quat_xyzw = tft.quaternion_from_matrix(T)

# Rearrange the quaternion to [w, x, y, z]
quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

print("Translation (x, y, z):", translation)
print("Quaternion (w, x, y, z):", quat_wxyz)
