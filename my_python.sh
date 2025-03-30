#!/usr/bin/env bash
while true; do
    echo "Running ycb_collection.py..."
    cd /home/chris/Chris/placement_ws/src/placement_quality
    /isaac-sim/python.sh -m ycb_simulation.ycb_collection --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
    echo "Script finished. Resting for 1 minute..."
    sleep 60
done