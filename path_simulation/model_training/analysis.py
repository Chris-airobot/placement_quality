import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from tqdm import tqdm

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def analyze_class_distribution(data, output_dir):
    print("[1/6] Analyzing class distributions…")
    succ = [s["success_label"] for s in data]
    coll = [s["collision_label"] for s in data]
    scount = Counter(succ)
    ccount = Counter(coll)
    # Save JSON
    with open(os.path.join(output_dir, "label_counts.json"), "w") as f:
        json.dump({"success": dict(scount), "collision": dict(ccount)}, f, indent=2)
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].bar(scount.keys(), scount.values(), color=['red','green'])
    axes[0].set_title("Success Label Distribution")
    axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["Fail","Success"])
    axes[1].bar(ccount.keys(), ccount.values(), color=['green','red'])
    axes[1].set_title("Collision Label Distribution")
    axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["No Collision","Collision"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "label_distribution.png"))
    plt.close(fig)
    print("    → Saved label_counts.json & label_distribution.png")

def analyze_pose_distributions(data, output_dir):
    print("[2/6] Analyzing position distributions…")
    init_z = np.array([s["initial_object_pose"][0] for s in data])
    final_pos = np.array([s["final_object_pose"][:3] for s in data])
    # Save JSON
    stats = {
        "init_z_mean": float(init_z.mean()), "init_z_std": float(init_z.std()),
        "final_x_mean": float(final_pos[:,0].mean()),
        "final_y_mean": float(final_pos[:,1].mean()),
        "final_z_mean": float(final_pos[:,2].mean()),
        "final_x_std":  float(final_pos[:,0].std()),
        "final_y_std":  float(final_pos[:,1].std()),
        "final_z_std":  float(final_pos[:,2].std()),
    }
    with open(os.path.join(output_dir, "position_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # Plot
    fig, axes = plt.subplots(2,1, figsize=(6,8))
    axes[0].hist(init_z, bins=50, color='blue', alpha=0.7)
    axes[0].set_title("Initial Z Distribution")
    axes[0].set_xlabel("Z")
    axes[1].hist(final_pos.flatten(), bins=50, color='orange', alpha=0.7)
    axes[1].set_title("Final XYZ Distribution (combined)")
    axes[1].set_xlabel("Value")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "position_distributions.png"))
    plt.close(fig)
    print("    → Saved position_stats.json & position_distributions.png")

def analyze_grasp_pose_variance(data, output_dir):
    print("[3/6] Analyzing grasp-pose orientation variance…")
    quats = np.array([s["grasp_pose"][3:] for s in data])
    norms = np.linalg.norm(quats, axis=1)
    # Save JSON
    stats = {
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
    }
    with open(os.path.join(output_dir, "grasp_quat_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # Plot histogram
    fig = plt.figure(figsize=(6,4))
    plt.hist(norms, bins=100, alpha=0.7)
    plt.title("Grasp Quaternion Norms")
    plt.savefig(os.path.join(output_dir, "grasp_quat_norms.png"))
    plt.close(fig)
    # PCA scatter
    pca = PCA(n_components=2).fit_transform(quats)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(pca[:,0], pca[:,1], alpha=0.2)
    plt.title("Grasp Orientation PCA")
    plt.savefig(os.path.join(output_dir, "grasp_quat_pca.png"))
    plt.close(fig)
    print("    → Saved grasp_quat_stats.json, grasp_quat_norms.png & grasp_quat_pca.png")

def analyze_orientation_change_vs_success(data, output_dir):
    print("[4/6] Computing orientation-change vs success…")
    angles = []
    labels = []
    for s in tqdm(data, desc="    Computing angles"):
        init = s["initial_object_pose"]
        r1 = R.from_quat([init[4], init[1], init[2], init[3]])
        # final always full
        r2 = R.from_quat([s["final_object_pose"][4], 
                          s["final_object_pose"][1], 
                          s["final_object_pose"][2], 
                          s["final_object_pose"][3]])
        angles.append((r1.inv() * r2).magnitude() * (180/np.pi))
        labels.append(s["success_label"])
    angles = np.array(angles); labels = np.array(labels)
    stats = {
        "mean_angle_success": float(angles[labels==1].mean()),
        "mean_angle_fail":    float(angles[labels==0].mean()),
    }
    with open(os.path.join(output_dir, "angle_change_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # Plot
    fig = plt.figure(figsize=(8,5))
    plt.hist(angles[labels==1], bins=50, alpha=0.6, label="Success")
    plt.hist(angles[labels==0], bins=50, alpha=0.6, label="Fail")
    plt.legend(); plt.title("Orientation Change vs Success")
    plt.xlabel("Angle (°)")
    plt.savefig(os.path.join(output_dir, "angle_vs_success.png"))
    plt.close(fig)
    print("    → Saved angle_change_stats.json & angle_vs_success.png")

def analyze_surface_transitions(data, output_dir):
    print("[5/6] Analyzing surface-to-surface transitions…")
    trans = [s["surfaces"] for s in data]
    counter = Counter(trans)
    with open(os.path.join(output_dir, "surface_counts.json"), "w") as f:
        json.dump(dict(counter), f, indent=2)
    matrix = np.zeros((6,6), dtype=int)
    for k,v in counter.items():
        i,j = map(int, k.split("_"))
        matrix[i,j] = v
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(matrix, origin='lower')
    ax.set_xlabel("Final surface"); ax.set_ylabel("Initial surface")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "surface_transitions.png"))
    plt.close(fig)
    print("    → Saved surface_counts.json & surface_transitions.png")

def visualize_random_samples(data, output_dir, num=5):
    print("[6/6] Logging random samples…")
    path = os.path.join(output_dir, "random_samples.txt")
    with open(path, "w") as f:
        for idx in np.random.choice(len(data), num, replace=False):
            s = data[idx]
            f.write(f"Index {idx} | grasp: {s['grasp_pose']} | init: {s['initial_object_pose']} | final: {s['final_object_pose']} | surface: {s['surfaces']} | success: {s['success_label']} | collision: {s['collision_label']}\n")
    print("    → Saved random_samples.txt")

def write_report(output_dir):
    report = []
    report.append("=== Data Analysis Report ===\n")
    # Load JSON stats
    with open(os.path.join(output_dir, "label_counts.json")) as f:
        lc = json.load(f)
    report.append("Class Distribution:\n")
    for k,v in lc["success"].items():
        report.append(f"  Success={k}: {v}\n")
    for k,v in lc["collision"].items():
        report.append(f"  Collision={k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "position_stats.json")) as f:
        ps = json.load(f)
    report.append("Position Statistics:\n")
    for k,v in ps.items():
        report.append(f"  {k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "grasp_quat_stats.json")) as f:
        gs = json.load(f)
    report.append("Grasp Quaternion Norms:\n")
    for k,v in gs.items():
        report.append(f"  {k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "angle_change_stats.json")) as f:
        ang = json.load(f)
    report.append("Orientation Change Stats:\n")
    for k,v in ang.items():
        report.append(f"  {k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "surface_counts.json")) as f:
        sc = json.load(f)
    report.append("Surface Transitions:\n")
    for k,v in sc.items():
        report.append(f"  {k}: {v}\n")
    report.append("\nRandom Samples:\n")
    with open(os.path.join(output_dir, "random_samples.txt")) as f:
        report.extend(f.readlines())
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.writelines(report)
    print(f"Analysis report saved to {os.path.join(output_dir, 'analysis_report.txt')}")

def data_analysis(data_path, output_dir):
    ensure_dir(output_dir)
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")

    analyze_class_distribution(data, output_dir)
    analyze_pose_distributions(data, output_dir)
    analyze_grasp_pose_variance(data, output_dir)
    analyze_orientation_change_vs_success(data, output_dir)
    analyze_surface_transitions(data, output_dir)
    visualize_random_samples(data, output_dir)
    write_report(output_dir)

if __name__ == "__main__":
    folder = "/media/chris/OS2/Users/24330/Desktop/placement_quality/unseen"
    data_path = os.path.join(folder, "all_data.json")
    output_path = os.path.join(folder, "analysis")

    data_analysis(data_path, output_path)