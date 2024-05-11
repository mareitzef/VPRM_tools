import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd

synmap_folder = "/home/madse/Downloads/VPRMpreproc_LCC_R118/"

# Open the GeoTIFF file
with rasterio.open(synmap_folder + "synmap.tif") as src:
    # Read the raster data
    data = src.read(1)

    # Get the metadata
    transform = src.transform
    crs = src.crs

    # Get the longitude and latitude values
    lons = np.arange(src.bounds.left, src.bounds.right, src.res[0])
    lats = np.arange(src.bounds.top, src.bounds.bottom, -src.res[1])

# TODO: read window corners from WRF inputfile

# Here: define the window for Europe
lon_min, lon_max = 5, 17
lat_min, lat_max = 44, 50

# Find the indices corresponding to the window
lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]
lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]

# Extract the window
data_window = data[
    lat_indices[0] : lat_indices[-1] + 1, lon_indices[0] : lon_indices[-1] + 1
]
lons_window = lons[lon_indices]
lats_window = lats[lat_indices]

# Create an xarray dataset
ds = xr.Dataset(
    {"synmap": (["lat", "lon"], data_window)},
    coords={"lon": lons_window, "lat": lats_window},
)
# Convert CRS to string
crs_str = crs.to_string()

# Add metadata
ds.attrs["crs"] = crs_str
ds.attrs["transform"] = transform

data_window = ds.synmap.values
lons_window = ds.lon.values
lats_window = ds.lat.values

# Create mesh grids of longitude and latitude
lon_mesh, lat_mesh = np.meshgrid(lons_window, lats_window)

# Define the mapping between SYNMAP and VPRM classes
mapping = {
    "VPRM": [
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        4,
        4,
        4,
        4,
        7,
        7,
        7,
        6,
        8,
        8,
        8,
        8,
    ],
    "Synmap": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
    ],
}

# Create DataFrame from the mapping
mapping_df = pd.DataFrame(mapping)

# Flatten the SYNMAP data
flattened_data = data_window.flatten()

# Create a DataFrame from the flattened SYNMAP data
synmap_df = pd.DataFrame(flattened_data, columns=["Synmap"])

# Merge the SYNMAP DataFrame with the mapping DataFrame to get the corresponding VPRM classes
result_df = pd.merge(synmap_df, mapping_df, on="Synmap", how="left")

# Check for missing mappings
missing_mappings = result_df[result_df["VPRM"].isnull()]

if not missing_mappings.empty:
    print("Warning: Missing mappings for the following SYNMAP values:")
    print(missing_mappings["Synmap"].unique())
    print("These values will not be included in the final result.")

# Fill missing values with a placeholder (e.g., -1)
result_df["VPRM"].fillna(-1, inplace=True)

# Now, reshape the result back to the original shape
result_array = result_df["VPRM"].values.reshape(data_window.shape)

# Convert result_array to an xarray DataArray and add it to the dataset
ds["VPRM"] = xr.DataArray(result_array, dims=("lat", "lon"))

# Remove the synmap variable from the dataset
ds = ds.drop_vars("synmap")

# Write dataset to NetCDF file
ds.to_netcdf("output.nc")

if plotting:
    # Define the legend labels
    legend_labels = {
        1: "Trees evergreen",
        2: "Trees deciduous",
        3: "Trees mixed",
        4: "Trees and shrubs",
        5: "Trees and grasses",
        6: "Trees and crops",
        7: "Grasses",
        8: "Barren, Urban and built-up, Permanent snow and ice",
    }

    # Create a plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Plot SYNMAP data
    mesh = ax.pcolormesh(
        lon_mesh, lat_mesh, result_array, transform=ccrs.PlateCarree(), cmap="viridis"
    )

    # Overlay European country borders
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")

    # Set plot title and labels
    ax.set_title("SYNMAP Data for Alps")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Create custom legend with colored markers
    legend_handles = []
    for key, value in legend_labels.items():
        color = plt.cm.viridis(key / len(legend_labels))
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=value,
                markerfacecolor=color,
                markersize=10,
                markeredgewidth=1,
            )
        )

    # Add legend below the plot
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=4
    )

    # Add colorbar with discrete steps by PFT
    cbar = plt.colorbar(
        mesh, ax=ax, shrink=0.5, aspect=10, ticks=np.arange(0.5, 8.5, 1)
    )
    cbar.set_ticks(np.arange(1, 9))
    cbar.set_ticklabels([str(i) for i in range(1, 9)])
    cbar.set_label("SYNMAP Data")

    plt.savefig(
        "SYNMAP_data_Alps.eps",
        dpi=300,
        bbox_inches="tight",
    )
    # Show the plot
    plt.show()


# used VPRM classes:
# 1 Trees evergreen
# 2 Trees deciduous
# 3 Trees mixed
# 4 Trees and shrubs
# 5 Trees and grasses
# 6 Trees and crops
# 7 grasses
# 8 Barren, Urban and built-up, Permanent snow and ice
# c("trees evergreen", "trees deciduous", "trees mixed", "trees and shrubs", "trees and grasses", "trees and crops", "grasses", "barren, urban and built-up, permanent snow and ice")

# SYNMAP classes:
# Trees Needle Evergreen Trees
# Trees Needle Deciduous
# Trees Needle Mixed
# Trees Broad Evergreen
# Trees Broad Deciduous
# Trees Broad Mixed
# Trees Mixed Evergreen
# Trees Mixed Deciduous
# Trees Mixed Mixed
# Trees and shrubs Needle Evergreen
# Trees and shrubs Needle Deciduous
# Trees and shrubs Needle Mixed
# Trees and shrubs Broad Evergreen
# Trees and shrubs Broad Deciduous
# Trees and shrubs Broad Mixed
# Trees and shrubs Mixed Evergreen
# Trees and shrubs Mixed Deciduous
# Trees and shrubs Mixed Mixed
# Trees and grasses Needle Evergreen
# Trees and grasses Needle Deciduous
# Trees and grasses Needle Mixed
# Trees and grasses Broad Evergreen
# Trees and grasses Broad Deciduous
# Trees and grasses Broad Mixed
# Trees and grasses Mixed Evergreen
# Trees and grasses Mixed Deciduous
# Trees and grasses Mixed Mixed
# Trees and crops Needle Evergreen Crops/natural vegetation mosaic
# Trees and crops Needle Deciduous
# Trees and crops Needle Mixed
# Trees and crops Broad Evergreen
# Trees and crops Broad Deciduous
# Trees and crops Broad Mixed
# Trees and crops Mixed Evergreen
# Trees and crops Mixed Deciduous
# Trees and crops Mixed Mixed
# Shrubs and crops
# Grasses and crops
# Crops
# Shrubs
# Shrubs and grasses
# Shrubs and barren
# Grasses
# Grasses and barren
# Barren
# Urban and built-up
# Permanent snow and ice

# MAPPING
# VPRM	Synmap
# 1	1
# 2	2
# 3	3
# 1	4
# 2	5
# 3	6
# 1	7
# 2	8
# 3	9
# 4	10
# 4	11
# 4	12
# 4	13
# 4	14
# 4	15
# 4	16
# 4	17
# 4	18
# 5	19
# 5	20
# 5	21
# 5	22
# 5	23
# 5	24
# 5	25
# 5	26
# 5	27
# 6	28
# 6	29
# 6	30
# 6	31
# 6	32
# 6	33
# 6	34
# 6	35
# 6	36
# 4	37
# 4	38
# 4	39
# 4	40
# 7	41
# 7	42
# 7	43
# 6	44
# 8	45
# 8	46
# 8	47
# 8	48
