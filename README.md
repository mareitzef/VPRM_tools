# VPRM_tools

#### get FLUXNET2015 data
download from https://fluxnet.org/data/fluxnet2015-dataset/

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

zip_filename="all_VPRM_optimized_params_diff_evo_V16_200.zip"
#### zip VPRM files
#zip -r $zip_filename $(find . -type f \( -name '*optimized_params_new_diff_evo_V2_100.xlsx' -o -name "*old*100.eps" -o -name "*check_input.eps" \) )
#### whithout plots but old an new
zip -r $zip_filename $(find . -type f \( -name '*optimized_params_*_diff_evo_V16_200.xlsx'  \) )

### local
#### download zip VPRM files from zip_filename=XYZ
cd Downloads/Fluxnet_Data/all_tuned_params/
scp c7071034@leo5.uibk.ac.at:/scratch/c7071034/DATA/Fluxnet2015/$zip_filename .
#### run script to unzip and move into folders
./unzip_and_mv $zip_filename 

#### then copy folders from Alps into Europe

### plot data with 
plots_for_VPRM_from_excel.ipynb
plot_VPRM_literature.ipynb

#### convert eps to png
plotname="IT-Tor_check_input"
gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pngalpha -r600 -sOutputFile=$plotname.png $plotname.eps

#### git stuff
##### delete local change e.g.:
git checkout -- submit_jobs_tune_VPRM.sh

ok