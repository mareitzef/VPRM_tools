#!/bin/bash

# Define the directory containing the folders
folder_directory="/home/madse/Downloads/Fluxnet_Data/Europe"

# List all subdirectories in the folder directory
subdirectories=$(find "$folder_directory" -mindepth 1 -maxdepth 1 -type d)

# Initialize an array to store incomplete folders
incomplete_folders=()

# Iterate over each subdirectory
for subdir in $subdirectories; do
  # Count the number of .xlsx files in the subdirectory
  num_xlsx_files=$(find "$subdir" -maxdepth 1 -name "*.xlsx" | wc -l)
  # Check if the number of .xlsx files is not equal to 12
  if [ "$num_xlsx_files" -ne 12 ]; then
    incomplete_folders+=("$subdir")
  fi
done

# Print incomplete folders
if [ ${#incomplete_folders[@]} -gt 0 ]; then
  echo "Incomplete folders:"
  printf '%s\n' "${incomplete_folders[@]}"
else
  echo "All folders are complete."
fi

