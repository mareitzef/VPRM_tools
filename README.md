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
zip -r VPRM_all_optimized_params_diff_evo_V2_100.zip $(find . -type f \( -name '*optimized_params_*_diff_evo_V2_100.xlsx' -o -name "*.eps" \) )
scp c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/VPRM_all_optimized_params_diff_evo_V2_100.zip .
