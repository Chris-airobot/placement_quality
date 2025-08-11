#!/bin/bash

# Configuration
EXPERIMENT_RESULTS_FILE="/home/chris/Chris/placement_ws/src/data/box_simulation/v4/experiments/experiment_results_test_data.jsonl"
SIMULATOR_FILE="/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/simulator.py"
ISAAC_SIM_PATH="/home/chris/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh"
SLEEP_TIME=10

# Function to get the latest index from experiment results and increment by 1
get_next_index() {
    if [ -f "$EXPERIMENT_RESULTS_FILE" ]; then
        # Get the last line and extract the index value
        last_line=$(tail -n 1 "$EXPERIMENT_RESULTS_FILE")
        if [ -n "$last_line" ]; then
            # Extract index value using a more robust method
            latest_index=$(echo "$last_line" | sed 's/.*"index": \([0-9]*\).*/\1/')
            # Increment by 1 to get the next index to process
            next_index=$((latest_index + 1))
            echo "$next_index"
        else
            echo "0"
        fi
    else
        echo "0"
    fi
}

# Function to update the data_index in simulator.py
update_data_index() {
    local new_index=$1
    local temp_file=$(mktemp)
    
    # Create backup
    cp "$SIMULATOR_FILE" "${SIMULATOR_FILE}.backup"
    
    # Update the data_index line
    sed "s/self\.data_index = [0-9]*/self.data_index = $new_index/" "$SIMULATOR_FILE" > "$temp_file"
    mv "$temp_file" "$SIMULATOR_FILE"
    
    echo "Updated data_index to $new_index in $SIMULATOR_FILE"
}

# Function to run the experiment
run_experiment() {
    echo "Running experiment with Isaac Sim..."
    $ISAAC_SIM_PATH -m placement_quality.cube_generalization.Experiment --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
    return $?
}

# Main execution
echo "=== Infinite Experiment Runner Script ==="
echo "Results file: $EXPERIMENT_RESULTS_FILE"
echo "Simulator file: $SIMULATOR_FILE"

# Initialize failure counter and last read index
failure_count=0
last_read_index=-1

# Run experiment infinitely
while true; do
    echo ""
    echo "=== Starting new experiment cycle ==="
    
    # Always read the latest index from results file
    current_latest_from_file=$(get_next_index)
    current_latest_actual=$((current_latest_from_file - 1))
    
    # Check if the results file has been updated since last read
    if [ $last_read_index -ne $current_latest_actual ]; then
        echo "Results file updated! Last read: $last_read_index, Current: $current_latest_actual"
        echo "Resetting failure count and using new index"
        failure_count=0
        current_index=$current_latest_from_file
        last_read_index=$current_latest_actual
        echo "Latest index from results: $current_latest_actual"
        echo "Next index to process: $current_index"
    else
        # Results file hasn't been updated, continue with current logic
        if [ $failure_count -eq 0 ]; then
            current_index=$current_latest_from_file
            echo "Latest index from results: $current_latest_actual"
            echo "Next index to process: $current_index"
        else
            echo "Continuing with failed index: $current_index"
        fi
    fi
    
    echo "Failure count: $failure_count"
    
    # Update the simulator file with the current index
    update_data_index $current_index
    
    echo "Running experiment with data_index: $current_index"
    
    # Run the experiment
    if run_experiment; then
        echo "✅ Experiment completed successfully!"
        echo "Waiting $SLEEP_TIME seconds before next experiment..."
        sleep $SLEEP_TIME
        
        # Reset failure counter on success
        failure_count=0
        
    else
        echo "❌ Experiment failed with exit code $?"
        failure_count=$((failure_count + 1))
        
        echo "Incrementing by $failure_count and retrying..."
        
        # Increment index by failure count for next attempt
        current_index=$((current_index + failure_count))
        update_data_index $current_index
        
        echo "Waiting $SLEEP_TIME seconds before retry with index $current_index..."
        sleep $SLEEP_TIME
    fi
done 