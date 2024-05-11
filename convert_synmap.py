import rasterio

# from rasterio.plot import show
import numpy as np

plotting = True
synmap_folder = "/home/madse/Downloads/VPRMpreproc_LCC_R118/"

# Import the dataset (consisting of unsigned integers)
D = np.fromfile(synmap_folder + "synmap_LC_jan1.bin", dtype=np.uint8)
# Reshape into the correct shape for SYNMAP: rows of length 43200 and columns of length 17500
D = D.reshape((17500, 43200))
# When interested, the associated longitude/latitude values are
lons = -180 + np.arange(43200) / 120
lats = 90 - np.arange(17500) / 120

# Write out as a GeoTIFF
with rasterio.Env():
    profile = profile = {
        "driver": "GTiff",
        "width": 43200,
        "height": 17500,
        "count": 1,
        "transform": rasterio.Affine(1 / 120, 0, -180, 0, -1 / 120, 90),
        # This transform is crucial, it maps the array-points
        # D[0,0] to the world location (lon0,lat0)=(-180,90)
        # D[1,0] to the world location (lon,lat)=(lon0,lat0)+(0,-1/120)
        # D[0,1] to the world location (lon,lat)=(lon0,lat0)+(+1/120,0)
        # Hence, it is what assigns locations to the array points
        "crs": rasterio.crs.CRS.from_dict(init="epsg:4326"),
        "tiled": True,
        "compress": "deflate",
        "dtype": rasterio.uint8,
    }
    with rasterio.open(synmap_folder + "synmap.tif", "w", **profile) as dst:
        dst.write(D.astype(rasterio.uint8), 1)
