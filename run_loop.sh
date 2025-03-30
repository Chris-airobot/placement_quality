#!/usr/bin/env bash


# A loop that runs indefinitely
while true
do
   
    echo "Running ycb_collection.py..."
    # Run the Isaac Sim Python script with your collection file
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh /home/chris/Chris/placement_ws/src/placement_quality/ycb_version/ycb_collection.py --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
    
    echo "Script finished. Resting for 3 minutes..."
    sleep 60  # 3 minutes
    
    # Loop repeats
done
