# VPRM_tools

## get FLUXNET2015 data

#### get MODIS timeseries by running Modis_timeseries_FluxNet.r in R
Rscript Modis_timeseries_FluxNet.r

#### copy the Fuxnet an modis data on the Cluster
zip FLX_AT-Mie_files.zip  FLX_AT-Mie_M* 
scp FLX_AT-Mie_files.zip c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/

#### run submit_jobs_tune_VPRM.sh for VPRM old and new
./submit_jobs_tune_VPRM.sh

### zip the results and download them 
### on LEO5
leo5
cdflx
zip_filename="Europe_VPRM_old_optimized_params_diff_evo_V2_100.zip"
#### zip VPRM files
zip -r $zip_filename $(find . -type f \( -name '*optimized_params_new_diff_evo_V2_100.xlsx' -o -name "*old*100.eps" -o -name "*check_input.eps" \) )
#### whithout plots
zip -r $zip_filename $(find . -type f \( -name '*optimized_params_old_diff_evo_V2_100.xlsx'  \) )

### local
#### download zip VPRM files
zip_filename="Europe_VPRM_old_optimized_params_diff_evo_V2_100.zip"
scp c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/$zip_filename .
unzip $zip_filename

then copy folders from Alps into Europe

### plot data with 
plots_for_VPRM_from_excel.ipynb
plot_VPRM_literature.ipynb

#### git stuff
##### delete local change e.g.:
git checkout -- submit_jobs_tune_VPRM.sh