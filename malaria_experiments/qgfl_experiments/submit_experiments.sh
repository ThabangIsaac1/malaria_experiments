#!/bin/bash
#
# Cluster Submission Script for Phase 1 Baseline Experiments
# Submits 6 YOLO baseline experiments (2 models × 3 datasets)
#
# Usage:
#   bash submit_experiments.sh
#
# Note: Modify SBATCH directives based on your cluster configuration
#

# Define experiment parameters
MODELS=("yolov8s" "yolov11s")
DATASETS=("d1" "d2" "d3")
TASK="binary"
STRATEGY="no_weights"
EPOCHS=200
BATCH_SIZE=16

# Cluster configuration (MODIFY FOR YOUR CLUSTER)
PARTITION="gpu"           # or your cluster's GPU partition name
TIME="48:00:00"          # 48 hours per job
MEM="32G"                # Memory per job
CPUS=8                   # CPU cores
GPUS=1                   # GPUs per job

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_baseline.py"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Create logs directory
mkdir -p "${LOGS_DIR}"

echo "=================================================="
echo "SUBMITTING PHASE 1 BASELINE EXPERIMENTS"
echo "=================================================="
echo "Task: ${TASK}"
echo "Strategy: ${STRATEGY}"
echo "Epochs: ${EPOCHS}"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Total jobs: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo "=================================================="
echo ""

# Counter for job numbering
job_count=0

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        ((job_count++))

        # Create unique job name
        job_name="E1.${job_count}_${model}_${dataset}_${TASK}_${STRATEGY}"

        # Log files
        log_out="${LOGS_DIR}/${job_name}.out"
        log_err="${LOGS_DIR}/${job_name}.err"

        echo "Submitting job ${job_count}/6: ${job_name}"

        # Create SLURM submission script
        cat <<EOF > "${LOGS_DIR}/${job_name}.sh"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --output=${log_out}
#SBATCH --error=${log_err}

# Environment setup
echo "=========================================="
echo "Job: ${job_name}"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "=========================================="

# Load modules (MODIFY FOR YOUR CLUSTER)
# module load python/3.11
# module load cuda/12.1
# module load cudnn/8.9

# Activate virtual environment (if using)
# source /path/to/venv/bin/activate

# Print GPU info
nvidia-smi

# Navigate to project directory
cd "${SCRIPT_DIR}"

# Run training
echo ""
echo "Starting training..."
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo "Task: ${TASK}"
echo "Strategy: ${STRATEGY}"
echo "Epochs: ${EPOCHS}"
echo ""

python train_baseline.py \\
    --model ${model} \\
    --dataset ${dataset} \\
    --task ${TASK} \\
    --strategy ${STRATEGY} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --workers ${CPUS} \\
    --use-wandb

echo ""
echo "=========================================="
echo "Job completed: \$(date)"
echo "=========================================="
EOF

        # Submit job
        # UNCOMMENT THE LINE BELOW FOR ACTUAL SUBMISSION
        # sbatch "${LOGS_DIR}/${job_name}.sh"

        # For testing, just show what would be submitted
        echo "  Created: ${LOGS_DIR}/${job_name}.sh"
        echo "  Logs: ${log_out}, ${log_err}"
        echo ""

    done
done

echo "=================================================="
echo "SUBMISSION COMPLETE"
echo "=================================================="
echo "Total jobs submitted: ${job_count}"
echo "Log directory: ${LOGS_DIR}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOGS_DIR}/*.out"
echo ""
echo "To actually submit jobs, uncomment the 'sbatch' line"
echo "in this script (line marked with UNCOMMENT)"
echo "=================================================="
