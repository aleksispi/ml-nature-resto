import numpy as np
from other_scripts.utils import read_image_and_polygon
import datetime
import geopandas as gpd
import time
import json
import xarray as xr
import netCDF4
import zarr
from numcodecs import Blosc
import os, sys
from paths import NC, GPKGS, ZARR

PATH_TO_GPKG = GPKGS/"final02-20.gpkg"
LOAD_PATH = NC
YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "year-after-data"]
START_IDX = 0
END_IDX = -1
IDXS_TO_USE = None

########## TEMPORARY UPDATE OF NAME TO NOT OVERWRITE ORIG
#ZARR = ZARR.with_name(ZARR.stem + "_aleksis.zarr")
###########

def _read_dates(filename):
    # Function used to get date-span of an nc-image time series
    im_nc = netCDF4.Dataset(filename)
    dates = netCDF4.num2date(im_nc.variables['t'], im_nc.variables['t'].units)
    dates = [date.strftime('%Y-%m-%d') for date in dates]
    return dates

# Read the GDF file containing polygons and matching year/areas
gdf = gpd.read_file(PATH_TO_GPKG)
gdf.head()
gdf.info()
gdf["Years_Areas"] = gdf["Years_Areas_json"].apply(json.loads)

# Extracting every unique date that occurs in the time-series
all_dates = []
print("Running loop over YEARS...")
for year in YEARS:

    # Create the full LOAD_PATH for the current year
    load_path = os.path.join(LOAD_PATH, year)
    if not os.path.exists(load_path):
        print(f"Error: The path {load_path} does not exist.")
        sys.exit()

    # List all the files in LOAD_PATH in name sorted order (note that the sorting criteria is
    # with respect to the integer at filename.split('/')[-1].split('_')[1]).
    files = [os.path.join(load_path, filepath) for filepath in os.listdir(load_path)]
    files = sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[1]))

    # Only keep the files that include the substring "10x10" and for which the 20x20 and 60x60 counterparts do exist
    files20 = [file for file in files if "20x20" in file]
    files60 = [file for file in files if "60x60" in file]
    files = [file for file in files if "10x10" in file]
    files = [file for file in files if file.replace("10x10", "20x20") in files20 and file.replace("10x10", "60x60") in files60]

    # Iterate over all the files
    if END_IDX == -1:
        end_idx = len(files)
    else:
        end_idx = END_IDX

    for file_idx in range(START_IDX, end_idx):
        print(year, file_idx)

        # Skip certain indices or activities
        if IDXS_TO_USE is not None and file_idx not in IDXS_TO_USE:
            continue

        filename = files[file_idx]
        dates = _read_dates(filename)
        dates = np.array(dates, dtype="datetime64[D]")
        all_dates.extend(dates)

# Count occurrences
unique_dates, counts = np.unique(all_dates, return_counts=True)
unique_dates = np.array(unique_dates, dtype="datetime64[D]")

# Taking these times and creating a time axis
time_to_idx = {t: i for i, t in enumerate(unique_dates)}

N_POLYGONS=len(gdf)

"""
Main zarr cube
The images are extracted as reflectance[p,t] and can be extracted for single polygons, arrays of them and same with time. Polygon indicies are stored in a GDF, each image has an image idx [p,t]
"""
print("Running main zarr cube steps...")
store = zarr.DirectoryStore(ZARR)
root = zarr.group(store, overwrite=True)
z = root.create_dataset(
    "reflectance",
    shape=(N_POLYGONS, len(unique_dates), 12, 100, 100),
    chunks=(1, 10, 12, 100, 100),
    dtype="int16",  # or float32
    fill_value=-9999,
    compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
)

# Side table: polygon-time -> image index
image_index = root.create_dataset(
    "image_index",
    shape=(N_POLYGONS, len(unique_dates)),
    chunks=(1, 10),
    dtype="int32",
    fill_value=-1,
)

# We will assign image_ids incrementally, x0 is smallest x value
image_x0 = root.create_dataset(
    "image_x0",
    shape=(0,),
    maxshape=(None,),
    dtype="float32",
    chunks=(1024,),
)

# y0 is larges y value as for images the first value is in the top
image_y0 = root.create_dataset(
    "image_y0",
    shape=(0,),
    maxshape=(None,),
    dtype="float32",
    chunks=(1024,),
)

# Store time axis once
time_stamps = root.create_dataset(
    "time_stamps",
    shape=(len(unique_dates),),
    dtype="datetime64[D]",
)

# How to connect an image idx to which Year/Area it originated from
image_file = root.create_dataset(
    "image_year",
    shape=(0,),
    maxshape=(None,),
    dtype="S16",
    chunks=(1024,),
)

time_stamps[:] = unique_dates.astype("datetime64[D]")

# Setting pixel size
root.attrs["pixel_size"] = 10.0

# Map filename → image_id so we don't duplicate x0/y0
image_registry = {}

"""
For each polygon extracting year/areas and then images from corresponding NC files
"""
print("Running main polygon loop...")
for polygon_idx in range(N_POLYGONS):
    print("Polygon: %d / %d" % (polygon_idx, N_POLYGONS))
    image_ids = gdf.loc[polygon_idx]["Years_Areas"]

    for image_id in image_ids:
        image_id_og=image_id
        image_id=image_id+"_10x10_image.nc"
        filename = os.path.join(LOAD_PATH,image_id)
        im, _, dates, y, x = read_image_and_polygon(filename)

        # im shape MUST be (H, W, C, T)
        H, W, C, T = im.shape
        assert H <= 100 and W <= 100, "Patch is larger than expected!"

        # Storing as int16. It is loss-less as it is the same as in sentinel
        im = im.astype("int16", copy=False)
        dates = np.array(dates, dtype="datetime64[D]")
        x0 = float(x[0])
        y0 = float(y[0])
        
        # Register image metadata
        if filename not in image_registry:
            # How many unique images do we have? That is the next image idx in line
            new_id = image_x0.shape[0]

            # Changing size to fit new metadata as we dont originally know size
            image_x0.resize(new_id + 1)
            image_y0.resize(new_id + 1)
            image_file.resize(new_id + 1)

            image_x0[new_id] = x0
            image_y0[new_id] = y0
            image_file[new_id] = image_id_og

            image_registry[filename] = new_id

        image_id_global = image_registry[filename]

        for t in range(T):
            date = dates[t]
            if date not in time_to_idx:
                continue

            time_idx = time_to_idx[date]
            image_index[polygon_idx, time_idx] = image_id_global

            # IMPORTANT: no coordinate-based sizing
            patch = np.full((100, 100, C), -9999, dtype=im.dtype)
            patch[:H, :W, :] = im[:, :, :, t]

            # write ONE chunk slice (hope this is one chunk) transposing to shape P x T x C x H x W
            z[polygon_idx, time_idx, :, :, :] = patch.transpose(2, 0, 1)