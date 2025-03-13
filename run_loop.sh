#!/usr/bin/env bash

# A loop that runs indefinitely
while true
do
    echo "Running random_collection.py..."
    python3 /home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/random_collection.py
    
    echo "Script finished. Resting for 3 minutes..."
    sleep 180  # 3 minutes
    
    # Loop repeats
done
