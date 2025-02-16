from learning_models.process_data_helpers import *
from learning_models.dataset import *





def encode_outputs(outputs: dict):
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

    # "values are:pose_diffs: 2.0033138697629886, ori_diffs: 2.9932579711083727, shift_poss: 0.13525934849764623, shift_oris: 1.6673673523277988, contacts: 5.0"
    pos_diff_max = 2.0033138697629886
    ori_diff_max = 2.9932579711083727
    shift_pos_max = 0.13525934849764623
    shift_ori_max = 1.6673673523277988
    contacts_max = 5.0

    # Convert Nones to 0.0 or some default
    # (Alternatively, you could skip these samples)
    if pos_diff is None: 
        pos_diff = pos_diff_max
    if ori_diff is None:
        ori_diff = ori_diff_max
    if shift_pos is None:
        shift_pos = shift_pos_max
    if shift_ori is None:
        shift_ori = shift_ori_max
    if contacts is None:
        contacts = contacts_max


    # Make them floats
    pos_diff  = float(pos_diff)
    ori_diff  = float(ori_diff)
    shift_pos = float(shift_pos)
    shift_ori = float(shift_ori)
    contacts  = float(contacts)

    
    params = {
        'pos_weight': 0.8,
        'shift_weight': 0.2,
        'conatct_weight': 0.2
    }

    stability_label = compute_stability_score(
        pos_diff, ori_diff, shift_pos, shift_ori, contacts,
        pos_diff_max, ori_diff_max, shift_pos_max, shift_ori_max, contacts_max,
        params=params
    )
    return feasibility_label, stability_label


    



def main():


    file_list = ["/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/Placement_22_False.json",
                 "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/Placement_27_False.json",
                 "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/Placement_70_False.json",
                 "/home/chris/Chris/placement_ws/src/random_data/run_20250214_203832/Grasping_4/Placement_4_False.json"]
    
    for file in file_list:
        data = process_file(file)
        cls, reg = encode_outputs(data["outputs"])
        print(f"This if file: {file}")
        print(f"Your classification label is: {cls}, and your regression label is: {reg}")
    

if __name__ == '__main__':
    main()
