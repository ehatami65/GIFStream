# --- Configuration ---
PROJECT_DIR="~/projects/GIFStream"
CONDA_ENV_NAME="GIFStream"

# The single scene directory to be processed.
SCENE_DIR="/data/shared/datasets/4DV/corgi-release"
RESULTS_BASE_DIR="/data/shared/aly/results/corgi"

# Output directories and tracking files will be located at the root of RESULTS_BASE_DIR.
AGGREGATE_CKPT_DIR="${RESULTS_BASE_DIR}/all_ckpts"
TRACKER_FILE="${RESULTS_BASE_DIR}/completed_gops.txt"
FAILED_LOG_FILE="${RESULTS_BASE_DIR}/failed_gops.txt"

# Python script for GIFStream and its static arguments
PYTHON_SCRIPT="examples/simple_trainer_GIFStream.py"
# The configuration type from the trainer script (e.g., neur3d_0, neur3d_1)
TRAINER_CONFIG_TYPE="neur3d_full"
STATIC_ARGS="--disable_viewer \
--save_steps 30000 \
--eval_steps 30000 \
--data_factor 2 \
--knn \
"

# --- Parallel Job Configuration ---
# Define the specific GPU IDs to be used for parallel jobs.
GPU_IDS=(2 3 4 5)
GOP_SIZE=60 # Each GOP will be 50 frames long

# --- Setup ---
# Expand the tilde (~) to the full home directory path
eval PROJECT_DIR="$PROJECT_DIR"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Error: Could not navigate to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# Initialize Conda for shell scripting
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate "$CONDA_ENV_NAME" || { echo "Error: Could not activate conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Activated Conda environment: $CONDA_ENV_NAME"
echo "---"

# Ensure the base directory and output directories/files exist
mkdir -p "$RESULTS_BASE_DIR"
mkdir -p "$AGGREGATE_CKPT_DIR"
touch "$TRACKER_FILE"
touch "$FAILED_LOG_FILE"

# Load already completed GOPs into an associative array for fast lookups
declare -A completed_gops_map
readarray -t completed_gops_list < "$TRACKER_FILE"
for gop in "${completed_gops_list[@]}"; do
    completed_gops_map["$gop"]=1
done

echo "Loaded ${#completed_gops_map[@]} completed GOPs to be skipped."
echo "---"

# --- Script Logic ---
# Create a queue of GOPs to process for the single scene
echo "Analyzing scene and breaking it into GOPs..."
GOP_QUEUE=()
scan_base_name=$(basename "$SCENE_DIR")

# Check if the directory contains an 'images' folder
if [ ! -d "${SCENE_DIR}/images" ]; then
    echo "Error: 'images' directory not found in '$SCENE_DIR'."
    exit 1
fi

# Find the first camera directory to count the frames
first_cam_dir=$(find "${SCENE_DIR}/images" -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$first_cam_dir" ]; then
    echo "Error: No camera directories found in '${SCENE_DIR}/images'."
    exit 1
fi

total_frames=$(find "$first_cam_dir" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
if [ "$total_frames" -eq 0 ]; then
    echo "Error: No images found in first camera directory '$first_cam_dir'."
    exit 1
fi

echo "-> Found scene '$scan_base_name' with $total_frames frames."

# Create a task for each GOP
for ((gop_start=0; gop_start < total_frames; gop_start+=GOP_SIZE)); do
    gop_id=$((gop_start / GOP_SIZE))
    gop_name="GOP_${gop_id}"

    if [[ -v completed_gops_map["$gop_name"] ]]; then
        echo "  -> Skipping GOP '$gop_name' (already completed)."
    else
        # Queue item format: "path/to/scan gop_start_frame total_frames_in_scan"
        GOP_QUEUE+=("$SCENE_DIR $gop_start $total_frames")
    fi
done

# Update TOTAL_JOBS to reflect only the ones we will run now
TOTAL_JOBS=${#GOP_QUEUE[@]}

if [ $TOTAL_JOBS -eq 0 ]; then
  echo "No new GOPs to process. All tasks are complete."
  exit 0
fi

echo "Found $TOTAL_JOBS new GOPs to process. Starting job queue..."
echo "---"

# Associative arrays to track job details by their Process ID (PID)
declare -A pids_to_gpu
declare -A pids_to_gop_name
declare -A pids_to_start_time
declare -A pids_to_result_dir

# Array to manage available GPUs
free_gpus=("${GPU_IDS[@]}")

# --- Main Loop ---
jobs_processed_count=0
total_duration=0

while [ $jobs_processed_count -lt $TOTAL_JOBS ]; do
  # Launch new jobs if there are free GPUs and GOPs in the queue
  while [ ${#free_gpus[@]} -gt 0 ] && [ ${#GOP_QUEUE[@]} -gt 0 ]; do
    # Get a free GPU and a GOP from the queues
    gpu_id=${free_gpus[0]}
    free_gpus=("${free_gpus[@]:1}") # Dequeue GPU
    
    # Dequeue GOP task and parse details
    read -r data_dir gop_start total_frames <<< "${GOP_QUEUE[0]}"
    GOP_QUEUE=("${GOP_QUEUE[@]:1}")

    # Calculate current GOP size to handle the last, possibly shorter, GOP
    remaining_frames=$((total_frames - gop_start))
    current_gop_size=$(( remaining_frames < GOP_SIZE ? remaining_frames : GOP_SIZE ))
    
    gop_id=$((gop_start / GOP_SIZE))
    gop_name="GOP_${gop_id}"
    
    # Results will be saved directly under the base results directory.
    result_dir="${RESULTS_BASE_DIR}/GOP_${gop_id}"
    rm -rf "$result_dir"
    mkdir -p "$result_dir"

    echo "🚀 Launching job for '$gop_name' on GPU $gpu_id..."
    LOG_FILE="${result_dir}/training.log"

    # Record start time and run the command in the background
    start_time=$(date +%s)
    
    (
        # This subshell ensures the command and its output redirection are managed as a single background process
        CUDA_VISIBLE_DEVICES=$gpu_id python $PYTHON_SCRIPT $TRAINER_CONFIG_TYPE $STATIC_ARGS \
            --data_dir "$data_dir" \
            --result_dir "$result_dir" \
            --start_frame "$gop_start" \
            --GOP_size "$current_gop_size"
    ) > "$LOG_FILE" 2>&1 &
    
    # Store the new job's PID and its associated data
    pid=$!
    pids_to_gpu[$pid]=$gpu_id
    pids_to_gop_name[$pid]=$gop_name
    pids_to_start_time[$pid]=$start_time
    pids_to_result_dir[$pid]=$result_dir
  done

  # Wait for any background job to finish and capture its exit code
  wait -n
  exit_code=$?
  
  # Now, find which job PID has finished since `wait -n` doesn't tell us
  finished_pid=""
  for pid in "${!pids_to_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
          finished_pid=$pid
          break
      fi
  done

  if [ -z "$finished_pid" ]; then
      # This can happen if all jobs finish at once at the end
      if [ ${#pids_to_gpu[@]} -eq 0 ]; then break; fi
      sleep 1; continue
  fi

  # --- Process the finished job ---
  # Retrieve job details
  gpu_id=${pids_to_gpu[$finished_pid]}
  gop_name=${pids_to_gop_name[$finished_pid]}
  start_time=${pids_to_start_time[$finished_pid]}
  result_dir=${pids_to_result_dir[$finished_pid]}
  
  # Calculate duration
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  total_duration=$((total_duration + duration))
  jobs_processed_count=$((jobs_processed_count + 1))

  # Check the exit code to determine success or failure
  if [ $exit_code -eq 0 ]; then
      # Success
      printf "✅ Job for '%s' (GPU %d) finished SUCCESSFULLY in %d min %d sec. (%d/%d complete)\n" \
          "$gop_name" "$gpu_id" "$((duration / 60))" "$((duration % 60))" "$jobs_processed_count" "$TOTAL_JOBS"
      
      # Add the GOP name to the tracker file on success
      echo "$gop_name" >> "$TRACKER_FILE"

      # Copy the final checkpoint to the aggregate directory
      echo "📦 Copying checkpoint for '$gop_name'..."
      source_ckpt_file="${result_dir}/ckpts/ckpt_29999_rank0.pt"
      if [ -f "$source_ckpt_file" ]; then
          destination_path="${AGGREGATE_CKPT_DIR}/${gop_name}.pt"
          cp "$source_ckpt_file" "$destination_path"
          echo "-> Copied checkpoint to $destination_path"
      else
          echo "⚠️  Warning: Checkpoint file not found for '$gop_name' at '$source_ckpt_file'."
      fi
  else
      # Failure
      printf "❌ Job for '%s' (GPU %d) FAILED with exit code %d after %d min %d sec. (%d/%d complete)\n" \
          "$gop_name" "$gpu_id" "$exit_code" "$((duration / 60))" "$((duration % 60))" "$jobs_processed_count" "$TOTAL_JOBS"
      
      # Log failures for later inspection
      echo "$gop_name (exit code: $exit_code)" >> "$FAILED_LOG_FILE"
  fi

  # Free up the GPU and remove the PID from tracking
  free_gpus+=($gpu_id)
  unset "pids_to_gpu[$finished_pid]"
  unset "pids_to_gop_name[$finished_pid]"
  unset "pids_to_start_time[$finished_pid]"
  unset "pids_to_result_dir[$finished_pid]"
done

# --- Final Report ---
echo "---"
echo "All $TOTAL_JOBS GOPs have been processed."
if [ $TOTAL_JOBS -gt 0 ]; then
    average_duration=$((total_duration / TOTAL_JOBS))
    printf "📊 Average job time: %d minutes and %d seconds.\n" "$((average_duration / 60))" "$((average_duration % 60))"
    printf "Total time: %d minutes and %d seconds.\n" "$((total_duration / 60))" "$((total_duration % 60))"
fi
# Deactivate conda environment
conda deactivate
echo "All tasks are complete." 