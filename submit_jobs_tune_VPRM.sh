#!/bin/bash

# Define variables
base_path="/scratch/c7071034/DATA/Fluxnet2015/Alps/"
maxiter=100  # (default=100 takes ages)
opt_method="diff_evo_V4"  # "diff_evo_V2"
VPRM_old_or_new="old"  # "old","new"

# List of folders
folders=($(find "$base_path" -type d -name "FLX_*"))

# Loop through each folder
for folder in "${folders[@]}"; do
    # Extract folder name from path
    folder_name=$(basename "$folder")
    # Create SLURM script for each job
    cat <<EOF >"job_${folder_name}_${VPRM_old_or_new}.sh"
#!/bin/bash
#SBATCH --job-name=tune_VPRM_${folder_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=360

module load python
srun python tune_VPRM.py -p "$base_path" -f "$folder_name" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new"
EOF

    # Submit the job to the cluster
    sbatch "job_${folder_name}_${VPRM_old_or_new}.sh"
done
