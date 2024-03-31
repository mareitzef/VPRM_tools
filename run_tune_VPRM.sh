#!/bin/bash

# Define variables
base_path="/home/madse/Downloads/Fluxnet_Data/"
maxiter=1  # (default=100 takes ages)
opt_method="diff_evo_V2"  # "minimize_V2","diff_evo_V2"
VPRM_old_or_new="new"  # "old","new"

# List of folders
folders=(
    # "FLX_AT-Mie_FLUXNET2015_FULLSET_2022-2022_1-4"
    # "FLX_AT-Neu_FLUXNET2015_FULLSET_2002-2012_1-4"
    # "FLX_CH-Cha_FLUXNET2015_FULLSET_2005-2014_2-4"
    # "FLX_CH-Dav_FLUXNET2015_FULLSET_1997-2014_1-4"
    # "FLX_CH-Fru_FLUXNET2015_FULLSET_2005-2014_2-4"
    # "FLX_CH-Lae_FLUXNET2015_FULLSET_2004-2014_1-4"
    # "FLX_CH-Oe1_FLUXNET2015_FULLSET_2002-2008_2-4"
    # "FLX_CH-Oe2_FLUXNET2015_FULLSET_2004-2014_1-4"
    # "FLX_CZ-wet_FLUXNET2015_FULLSET_2006-2014_1-4"
    # "FLX_DE-Lkb_FLUXNET2015_FULLSET_2009-2013_1-4"
    # "FLX_DE-SfN_FLUXNET2015_FULLSET_2012-2014_1-4"
     "FLX_ES-Ln2_FLUXNET2015_FULLSET_2009-2009_1-4"
    # "FLX_IT-Isp_FLUXNET2015_FULLSET_2013-2014_1-4"
    # "FLX_IT-La2_FLUXNET2015_FULLSET_2000-2002_1-4"
    # "FLX_IT-Lav_FLUXNET2015_FULLSET_2003-2014_2-4"
    # "FLX_IT-MBo_FLUXNET2015_FULLSET_2003-2013_1-4"
    # "FLX_IT-PT1_FLUXNET2015_FULLSET_2002-2004_1-4"
    # "FLX_IT-Ren_FLUXNET2015_FULLSET_1998-2013_1-4"
    # "FLX_IT-Tor_FLUXNET2015_FULLSET_2008-2014_2-4"
)

# Loop through each folder
for folder in "${folders[@]}"; do
    # Call Python script with variables as command line arguments
    python tune_VPRM.py -p "$base_path" -f "$folder" -i "$maxiter" -m "$opt_method" -v "$VPRM_old_or_new"
done

