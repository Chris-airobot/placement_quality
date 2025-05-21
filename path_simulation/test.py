def mesh_saving():
    import os
    import numpy as np
    from pxr import Usd, UsdGeom
    import omni

    # Example paths
    input_mesh_path = "/World/_09_gelatin_box/_09_gelatin_box"  # Replace with actual input path
    output_mesh_path = "/home/chris/Desktop/mesh.obj"  # Replace with desired output path

    # Create a new stage
    stage = omni.usd.get_context().get_stage()

    # Load the mesh to the stage
    mesh_prim_path = "/World/_09_gelatin_box"

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)

    # Get the mesh prim
    mesh_prim = UsdGeom.Mesh(stage.GetPrimAtPath(input_mesh_path))
    if not mesh_prim:
        print(f"No mesh found at {input_mesh_path}")
        exit(1)
    print(mesh_prim)

    # Extract mesh data
    points = mesh_prim.GetPointsAttr().Get()
    face_vertex_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()

    # Convert data for OBJ export
    vertices = np.array(points) * 1000
    face_data = []
    idx = 0
    for count in face_vertex_counts:
        if count == 3:  # Triangle
            face_data.append([face_vertex_indices[idx], face_vertex_indices[idx+1], face_vertex_indices[idx+2]])
        elif count == 4:  # Quad
            face_data.append([face_vertex_indices[idx], face_vertex_indices[idx+1], face_vertex_indices[idx+2], face_vertex_indices[idx+3]])
        idx += count

    # Save as OBJ file manually
    print(f"Saving mesh to {output_mesh_path}")
    with open(output_mesh_path, 'w') as f:
        # Write header/comment
        f.write("# OBJ file created from Isaac Sim\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces (OBJ indices are 1-based)
        for face in face_data:
            f.write("f " + " ".join([str(idx + 1) for idx in face]) + "\n")
            
    print(f"Mesh saved successfully to {output_mesh_path}")



def convert_mesh():
    import trimesh
    import numpy as np
    mesh = trimesh.load('/home/chris/Desktop/mesh_meter.obj')
    mesh.show()
    mesh.export('/home/chris/Desktop/gelatin_box_meter.ply')  # Inventor (.ply) file





def self_implemntation():
    import trimesh
    import numpy as np

    # ---- CONFIG ----
    mesh_path = "/home/chris/Desktop/gelatin_box_meter.ply"  # your mesh file
    n_samples = 500   # how many surface points to sample (change as needed)
    max_width = 0.08  # gripper max opening in meters
    min_width = 0.01  # gripper min opening in meters
    angle_thresh_deg = 20  # antipodal normal angle threshold in degrees

    # ---- LOAD MESH ----
    mesh = trimesh.load(mesh_path)
    points, face_indices = trimesh.sample.sample_surface_even(mesh, n_samples)
    normals = mesh.face_normals[face_indices]

    grasps = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p1, n1 = points[i], normals[i]
            p2, n2 = points[j], normals[j]
            width = np.linalg.norm(p1 - p2)
            if not (min_width < width < max_width):
                continue
            # antipodal check: normals roughly opposed
            angle = np.degrees(np.arccos(np.clip(np.dot(n1, -n2), -1.0, 1.0)))
            if angle > angle_thresh_deg:
                continue
            # Center position and gripper axis
            center = (p1 + p2) / 2
            x_axis = (p2 - p1) / width
            z_axis = np.cross(x_axis, [0, 0, 1])
            if np.linalg.norm(z_axis) < 1e-3:
                z_axis = np.cross(x_axis, [0, 1, 0])
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            R = np.column_stack((x_axis, y_axis, z_axis))  # Rotation matrix: x=grip axis
            grasps.append({"position": center, "orientation": R, "width": width})

    print(f"Generated {len(grasps)} antipodal grasps!")

    # ---- SAVE TO FILE ----
    # import pickle
    # with open("grasp_candidates.pkl", "wb") as f:
    #     pickle.dump(grasps, f)
    # print("Saved grasps to grasp_candidates.pkl")





if __name__ == "__main__":
    # convert_mesh()
    self_implemntation()