import torch
from torch.utils.data import Dataset
import numpy as np

def compute_stability_score(pos_diff, ori_diff, shift_pos, shift_ori, contacts,
                            pos_max, ori_max, shift_pos_max, shift_ori_max, contacts_max, params=None):
    """
    Computes a stability score in [0,1] by first normalizing each metric to [0,1]
    using the provided upper bounds, then combining them with weighted penalties.
    """
    pos_norm       = pos_diff / pos_max
    ori_norm       = ori_diff / ori_max
    shift_pos_norm = shift_pos / shift_pos_max
    shift_ori_norm = shift_ori / shift_ori_max
    contacts_norm  = contacts / contacts_max

    penalty = (params['pos_weight'] * pos_norm + 
               params['pos_weight'] * ori_norm + 
               params['shift_weight'] * shift_pos_norm + 
               params['shift_weight'] * shift_ori_norm + 
               params['conatct_weight'] * contacts_norm)
    
    stability = 100 - penalty  # No clamping is applied here
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
            # Encode inputs
            x = self.encode_inputs(entry["inputs"])
            # Encode outputs as a tuple: (classification label, regression label)
            labels = self.encode_outputs(entry["outputs"])
            if x is not None and labels is not None:
                self.samples.append((x, labels))

    def encode_inputs(self, inputs):
        """
        Flatten and concatenate input features into a 1D tensor.
        """
        init_pos = np.array(inputs["cube_initial_position"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_initial_rel_position"], dtype=np.float32)
        init_ori = np.array(inputs["cube_initial_orientation"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_initial_rel_orientation"], dtype=np.float32)
        targ_pos = np.array(inputs["cube_target_position"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_target_rel_position"], dtype=np.float32)
        targ_ori = np.array(inputs["cube_target_orientation"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_target_rel_orientation"], dtype=np.float32)
        grasp_pos = np.array(inputs["grasp_position"], dtype=np.float32) if inputs.get("grasp_position") is not None else init_pos
        grasp_ori = np.array(inputs["grasp_orientation"], dtype=np.float32) if inputs.get("grasp_orientation") is not None else init_ori

        x_concat = np.concatenate([
            grasp_pos, grasp_ori,
            init_pos, init_ori,
            targ_pos, targ_ori
        ])
        return torch.from_numpy(x_concat)

    def encode_outputs(self, outputs: dict):
        """
        Encode outputs into two labels:
          - Classification: feasibility (0 for failure, 1 for success)
          - Regression: stability score in [0, 1]
        """
        is_fail = outputs.get("grasp_unsuccessful", False) or outputs.get("bad", False)
        feasibility_label = 0 if is_fail else 1

        pos_diff_max = 0.6081497916935493
        ori_diff_max = 2.848595486712802
        shift_pos_max = 0.0794796857415393
        shift_ori_max = 2.1095218360699306
        contacts_max = 4.0

        pos_diff = outputs.get("position_difference", pos_diff_max)
        ori_diff = outputs.get("orientation_difference", ori_diff_max)
        shift_pos = outputs.get("shift_position", shift_pos_max)
        shift_ori = outputs.get("shift_orientation", shift_ori_max)
        contacts = outputs.get("contacts", contacts_max)

        pos_diff = float(pos_diff)
        ori_diff = float(ori_diff)
        shift_pos = float(shift_pos)
        shift_ori = float(shift_ori)
        contacts = float(contacts)

        

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
        return (feasibility_label, stability_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, (cls_lbl, reg_lbl) = self.samples[idx]
        cls_lbl = torch.tensor(cls_lbl, dtype=torch.float32)
        reg_lbl = torch.tensor(reg_lbl, dtype=torch.float32)
        return x, (cls_lbl, reg_lbl)
