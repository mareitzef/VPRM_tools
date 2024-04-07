#!/bin/bash

# Define variables
base_path="/home/madse/Downloads/Fluxnet_Data/Europe/"
maxiter=1  # (default=100 takes ages)
opt_method="diff_evo_V2"  # "minimize_V2","diff_evo_V2"
VPRM_old_or_new="new"  # "old","new"

# List of folders
folders=(
"FLX_DE-Hai_FLUXNET2015_FULLSET_2000-2012_1-4"
"FLX_RU-Fyo_FLUXNET2015_FULLSET_1998-2014_2-4"
)

# Loop through each folder
for folder in "${folders[@]}"; do
    # Call Python script with variables as command line arguments
    python tune_VPRM.py -p "$base_path" -f "$folder" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new"
done

