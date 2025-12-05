#!/usr/bin/env bash
set -e

python3 /home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/data_processing.py
python3 /home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/dataset.py
python3 /home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/model_training.py


