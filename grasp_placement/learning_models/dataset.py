import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


def compute_stability_score(pos_diff, ori_diff, shift_pos, shift_ori, contacts,
                            pos_max=0.0, 
                            ori_max=0.0, 
                            shift_pos_max=0.0, 
                            shift_ori_max=0.0, 
                            contacts_max=0.0, params=None):
    """
    Computes a stability score in [0,1] by first normalizing each metric to [0,1]
    using the provided upper bounds, then combining them with weighted penalties.
    
    Parameters:
      pos_diff:         The position difference error.
      ori_diff:         The orientation difference error.
      shift_pos:        The shift in position during settling.
      shift_ori:        The shift in orientation during settling.
      contacts:         The number of contacts.
      
      pos_max:          The upper bound for pos_diff (e.g., 90th percentile).
      ori_max:          The upper bound for ori_diff.
      shift_pos_max:    The upper bound for shift_pos.
      shift_ori_max:    The upper bound for shift_ori.
      contacts_max:     The upper bound for contacts.
      
    Returns:
      stability:        A score between 0 and 1 where 1 is best.
    """
    # Normalize each metric to the range [0,1]
    pos_norm     = min(pos_diff / pos_max, 1.0)
    ori_norm     = min(ori_diff / ori_max, 1.0)
    shift_pos_norm = min(shift_pos / shift_pos_max, 1.0)
    shift_ori_norm = min(shift_ori / shift_ori_max, 1.0)
    contacts_norm  = min(contacts / contacts_max, 1.0)

    # For each metric, choose a weight based on whether it exceeds its threshold
    # pos_weight = params['h_diff_weight'] if pos_diff > 0.1 else params['pos_weight']
    # ori_weight = params['h_diff_weight'] if ori_diff > ori_threshold else params['pos_weight']
    # shift_pos_weight = params['h_shift_weight'] if shift_pos > shift_pos_threshold else params['shift_weight']
    # shift_ori_weight = params['h_shift_weight'] if shift_ori > shift_ori_threshold else params['shift_weight']
    # contacts_weight = params['h_contact_weight'] if contacts > contacts_threshold else params['conatct_weight']




    # Combine the normalized errors using chosen weights.
    penalty = (params['pos_weight'] * pos_norm + 
               params['pos_weight'] * ori_norm + 
               params['shift_weight'] * shift_pos_norm + 
               params['shift_weight'] * shift_ori_norm + 
               params['conatct_weight'] * contacts_norm)
    
    # Subtract the penalty from a perfect score (1.0)
    stability = 1.0 - penalty

    # Clamp the result to [0,1]
    stability = max(0.0, min(stability, 1.0))
    return stability



class MyStabilityDataset(Dataset):
    def __init__(self, all_entries):
        """
        all_entries is a list of dicts, each with:
          - "inputs": { ... }  
          - "outputs": { ... }
        """
        self.samples = []
        self.gripper_reference = False

        for entry in all_entries:
            # 1) Encode the inputs as a numeric vector
            x = self.encode_inputs(entry["inputs"])

            # Multi-head label or single dictionary
            # Example: (feasibility, final_pos_err, final_ori_err, delta_pos, delta_ori)
            feas_lbl, reg_lbl = self.encode_outputs(entry["outputs"])

              # If you want to skip any incomplete data, do a check:
            if x is not None and feas_lbl is not None and reg_lbl is not None:
                self.samples.append((x, feas_lbl, reg_lbl))


    def encode_inputs(self, inputs):
        """
        Flatten or concatenate everything into a 1D tensor.
        E.g., positions/orientations are lists or arrays.
        """
        # print(f"encode_inputs: {inputs}")
        
        

        init_pos = np.array(inputs["cube_initial_position"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_initial_rel_position"], dtype=np.float32)
        init_ori = np.array(inputs["cube_initial_orientation"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_initial_rel_orientation"], dtype=np.float32)

        targ_pos = np.array(inputs["cube_target_position"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_target_rel_position"], dtype=np.float32)
        targ_ori = np.array(inputs["cube_target_orientation"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_target_rel_orientation"], dtype=np.float32)

        grasp_pos = np.array(inputs["grasp_position"], dtype=np.float32) if inputs.get("grasp_position") else init_pos
        grasp_ori = np.array(inputs["grasp_orientation"], dtype=np.float32) if inputs.get("grasp_orientation") else init_ori

        
        x_concat = np.concatenate([
            grasp_pos, grasp_ori,
            init_pos, init_ori,
            targ_pos, targ_ori
        ])
        return torch.from_numpy(x_concat)


    def encode_outputs(self, outputs: dict):
        """
        - Head A (classification): feasibility = 0 or 1
            If `grasp_unsuccessful` OR `bad` is True => 0 (fail), else 1 (success).
        - Head B (regression): [pose_diff, ori_diff, shift_pos, shift_ori, contacts]
        """
        # Classification (feasibility)
        is_fail = (outputs.get("grasp_unsuccessful", False) or 
                   outputs.get("bad", False))
        feasibility_label = 0 if is_fail else 1  # 0=fail, 1=success

        pos_diff = outputs.get("position_difference", None)
        ori_diff = outputs.get("orientation_difference", None)
        shift_pos = outputs.get("shift_position", None)
        shift_ori = outputs.get("shift_orientation", None)
        contacts = outputs.get("contacts", None)

        # Convert Nones to 0.0 or some default
        # (Alternatively, you could skip these samples)
        if pos_diff is None: 
            pos_diff = 0.0
        if ori_diff is None:
            ori_diff = 0.0
        if shift_pos is None:
            shift_pos = 0.0
        if shift_ori is None:
            shift_ori = 0.0
        if contacts is None:
            contacts = 0


        # Make them floats
        pos_diff  = float(pos_diff)
        ori_diff  = float(ori_diff)
        shift_pos = float(shift_pos)
        shift_ori = float(shift_ori)
        contacts  = float(contacts)

        # "values are:pose_diffs: 2.0033138697629886, ori_diffs: 2.9932579711083727, shift_poss: 0.13525934849764623, shift_oris: 1.6673673523277988, contacts: 5.0"
        pos_diff_max = 2.0033138697629886
        ori_diff_max = 2.9932579711083727
        shift_pos_max = 0.13525934849764623
        shift_ori_max = 1.6673673523277988
        contacts_max = 5.0

        params = {
            'h_diff_weight': 0.6,
            'pos_weight': 0.3,
            'h_shift_weight': 0.2,
            'shift_weight': 0.1,
            'h_contact_weight': 0.8,
            'conatct_weight': 0.4
        }

        stability_label = compute_stability_score(
            pos_diff, ori_diff, shift_pos, shift_ori, contacts,
            pos_diff_max, ori_diff_max, shift_pos_max, shift_ori_max, contacts_max,
            params=params
        )
        return feasibility_label, stability_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, feas_lbl, reg_lbl = self.samples[idx]
        # Classification label as a single int
        feas_lbl = torch.tensor(feas_lbl, dtype=torch.long)  # needed for cross-entropy
        return x, (feas_lbl, reg_lbl)