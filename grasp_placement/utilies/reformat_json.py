import json

# Example JSON string
file_path = "/home/chris/Chris/placement_ws/src/grasp_placement/data/Grasping_0/Placement_0.json"

# Parse the JSON string into a Python dictionary

with open(file_path, "r") as file:
    raw_data = json.load(file)  # Parse JSON into a Python dictionary



# Pretty-print the JSON
pretty_json = json.dumps(raw_data, indent=4)


with open("/home/chris/Chris/placement_ws/src/grasp_placement/data/Grasping_0/Placement_0_pretty.json", "w") as pretty_file:
    pretty_file.write(pretty_json)
