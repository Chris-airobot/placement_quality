import torch
from torch.utils.data import Dataset

class MyStabilityDataset(Dataset):
    def __init__(self, all_entries):
        """
        all_entries is a list of dicts, each with:
          - "inputs": { ... }  
          - "outputs": { ... }
        """
        self.samples = []

        for entry in all_entries:
            # 1) Encode the inputs as a numeric vector
            x = self.encode_inputs(entry["inputs"])
            # 2) Compute the 0->1 stability label from outputs
            y = compute_stability_score(entry["outputs"])
            self.samples.append((x, y))

    def encode_inputs(self, inputs):
        """
        Flatten or concatenate everything into a 1D tensor.
        Example: positions/orientations are lists or arrays.
        """
        import numpy as np

        # Extract everything into floats
        grasp_pos = np.array(inputs["Grasp_position"], dtype=np.float32)
        grasp_ori = np.array(inputs["Grasp_orientation"], dtype=np.float32)
        init_pos = np.array(inputs["cube_initial_position"], dtype=np.float32)
        init_ori = np.array(inputs["cube_initial_orientation"], dtype=np.float32)
        targ_pos = np.array(inputs["cube_target_position"], dtype=np.float32)
        targ_ori = np.array(inputs["cube_target_orientation"], dtype=np.float32)
        targ_surf = np.array([inputs["cube_target_surface"]], dtype=np.float32)

        x_concat = np.concatenate([
            grasp_pos, grasp_ori,
            init_pos, init_ori,
            targ_pos, targ_ori,
            targ_surf
        ])
        return torch.from_numpy(x_concat)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, torch.tensor([y], dtype=torch.float32)  # shape [1]


def compute_stability_score(outputs):
    """
    Convert the final outputs into a single 0->1 stability score.
    Higher => better (more stable/accurate).
    """

    if outputs["grasp_unsuccessful"]:
        return -100
    
    
    pos_error = outputs["pose_shift_position"]      # e.g. ~0.04 m
    ori_error = outputs["pose_shift_orientation"]   # e.g. ~1.5 rad
    unsuccessful_grasp = outputs["grasp_unsuccessful"]
    placement_position_differece = outputs["position_differece"]
    placement_orientation_differece = outputs["orientation_differece"]
    placement_position_ok = outputs["position_successful"]
    placement_orientation_ok = outputs["orientation_successful"]


    # Weights for typical scale:
    b_pos = -20.0  # so that 0.01 m error contributes about 0.1
    b_ori = -10.0   # so that 1 rad orientation error contributes 1.0

    p_pos = -25.0
    p_ori = -10.0
    error_score = b_pos * pos_error + b_ori * ori_error 
    # error_score = b_pos * pos_error + b_ori * ori_error + p_pos * placement_position_differece + p_ori * placement_orientation_differece



    boolean_score = 0.0
    if not unsuccessful_grasp:
        boolean_score += 50.0  # Grasp success gives a significant boost
    if placement_position_ok:
        boolean_score += 30.0  # Placement position success contributes
    if placement_orientation_ok:
        boolean_score += 30.0  # Placement orientation success contributes


    # Convert total_error to [0..1] score:
    # if total_error = 0 => perfect => score=1
    # if total_error >=1 => score=0
    raw_score = error_score + boolean_score

    # min_score = -200
    # max_score = 50

    min_score = -100
    max_score = 110

    final_score = (raw_score - min_score) / (max_score - min_score)

    return final_score