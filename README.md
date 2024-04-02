# VPRM_tools

# get FLUXNET2015 data

# get MODIS timeseries by running Modis_timeseries_FluxNet.r in R
Rscript Modis_timeseries_FluxNet.r

# copy the Fuxnet an modis data on the Cluster
zip AT-Mie_files.zip  AT-Mie_M* 
scp AT-Mie_files.zip c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/FLX_AT-Mie_FLUXNET2015_FULLSET_2022-2022_1-4/

# run submit_jobs_tune_VPRM.sh for VPRM old and new
./submit_jobs_tune_VPRM.sh

# zip the results and download them 
# VPRM old
zip -r VPRM_old_optimized_params_diff_evo_V2_100.zip $(find . -type f \( -name '*optimized_params_old_diff_evo_V2_100.xlsx' -o -name "*old*.eps" -o -name "*check_input.eps" \) )
scp c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/VPRM_old_optimized_params_diff_evo_V2_100.zip .
# VPRM new
zip -r VPRM_new_optimized_params_diff_evo_V2_100.zip $(find . -type f \( -name '*optimized_params_new_diff_evo_V2_100.xlsx' -o -name "*new*.eps" -o -name "*check_input.eps" \) )
scp c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/VPRM_new_optimized_params_diff_evo_V2_100.zip .

# plot data with 
plots_for_VPRM_from_excel.ipynb

# git stuff
# delete local change e.g.:
git checkout -- submit_jobs_tune_VPRM.sh