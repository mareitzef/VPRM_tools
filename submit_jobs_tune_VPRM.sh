#!/bin/bash

# Define variables
base_paths=(
    "/scratch/c7071034/DATA/Fluxnet2015/Alps/"
)   # "/scratch/c7071034/DATA/Fluxnet2015/Europe/"
maxiter=100  
opt_method="diff_evo_V19"  # method and version
VPRM_options=("migli" "old") # "migli" "new" "old"

# Loop through each base path
for base_path in "${base_paths[@]}"; do
    # Loop through each VPRM option
    for VPRM_old_or_new in "${VPRM_options[@]}"; do
        # List of folders
        folders=($(find "$base_path" -type d -name "FLX_*"))

        # Loop through each folder
        for folder in "${folders[@]}"; do
            # Extract folder name from path
            folder_name=$(basename "$folder")
            # Create SLURM script for each job
            cat <<EOF >"job_${folder_name}_${VPRM_old_or_new}.sh"
#!/bin/bash
#SBATCH --job-name=tune_VPRM_${folder_name}_${VPRM_old_or_new}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=120

module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "\$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/pyrealm
srun python tune_VPRM.py -p "$base_path" -f "$folder_name" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new"
EOF

            # Submit the job to the cluster
            sbatch "job_${folder_name}_${VPRM_old_or_new}.sh"
        done
    done
done
