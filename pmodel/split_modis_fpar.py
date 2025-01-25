import xarray as xr
import os

# Path to the original MODIS NetCDF file
modis_path = "/home/madse/Downloads/MODIS_data/MCD15A3H.061_500m_aid0001.nc"

# Output directory to save the split files
output_dir = "/home/madse/Downloads/MODIS_data/split_timesteps"
os.makedirs(output_dir, exist_ok=True)

# Open the NetCDF dataset
modis_ds = xr.open_dataset(modis_path)

# Iterate through the time dimension
for t_idx, t_value in enumerate(modis_ds["time"]):
    # Select the data for the current timestep
    timestep_ds = modis_ds.isel(time=t_idx)

    # Convert cftime.DatetimeJulian to a standard ISO 8601 date string
    date_str = str(t_value.dt.strftime("%Y%m%d").item())

    # Create the output file path
    output_path = os.path.join(output_dir, f"MODIS_fpar_{date_str}.nc")

    # Save the subset dataset
    timestep_ds.to_netcdf(output_path, format="NETCDF3_CLASSIC")

    print(f"Saved timestep {t_idx + 1}/{len(modis_ds['time'])} to {output_path}")

print("All timesteps saved.")
