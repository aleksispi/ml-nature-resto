import os, sys
import openeo
import argparse
import shutil
import numpy as np
from utils import create_square_bounding_box, time_measurement, get_layer_data_from_gpkg

# Add argument parser for reading the arguments POLY_IDX_START and POLY_IDX_END
# from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--POLY_IDX_START", type=int, default=0, help="Index of first polygon to process")
parser.add_argument("--POLY_IDX_END", type=int, default=-1, help="End index of polygons to process (exclusive)")
args = parser.parse_args()


# Global vars
CONNECTION = 'des'  # 'des' (Digital Earth Sweden) or 'cop' (Copernicus)
COLLECTION = 'SENTINEL2_L2A' # 'SENTINEL2_L2A' only possible
MARGIN_PER_SIDE = 0.85  # Margin around polygon when extracting bounding box (if negative, means percentage of polygon size, otherwise this specifies the size in absolut terms, so each polygon gets a same-sized image to it)
#GPKG_PATH = "../data-from-jv/joined_allhist_with_arslager.gpkg" #joined_curr_with_arslager.gpkg"  # Path to GPKG file ´
#GPKG_PATH = "../data-from-jv/georg-preprocessed/arslager/Year-2019.gpkg"
GPKG_PATH = "../data-from-jv/georg-preprocessed/Restored_2-7Years_startafter2017.gpkg"
POLY_IDX_START = args.POLY_IDX_START  # Index of first polygon to process
POLY_IDX_END = args.POLY_IDX_END  # End index of polygons to process (exclusive)
SAVE_PATH = "../sen2a-data-mark-georg/year-after-data/"
YEAR_SPAN = ['2019', '2025']  # Years to consider for the data
MONTH_DAY_SPAN = ['06-01', '08-31']#['06-01', '06-30']  # Dates to consider for the data
DATE_SPANS = [[f"{year}-{month_day}" for month_day in MONTH_DAY_SPAN] for year in YEAR_SPAN]
### TODO WILL NOT DO THIS IN THE END ####
DATE_SPAN = DATE_SPANS[0]  # Use the first date span for now
####

# Extract polygons from the GPKG file, if file is of type .gpkg.
# Otherwise, if .npy, the below is assumed to have been done already.
if GPKG_PATH.endswith(".gpkg"):
    layer_data, category_exists = get_layer_data_from_gpkg(GPKG_PATH, keep_only_unique_ids=True)
    # Assert that the GPKG has exactly one layer
    assert len(layer_data) == 1, "The GPKG file should have exactly one layer."
    # Extract that singular layer
    layer_data = layer_data[list(layer_data.keys())[0]]
    geometries = layer_data['polys']
    categories = layer_data['category']
    timestamps = layer_data['timestamps']
else:
    print("FIX ERROR, NOT IMPLEMENTED YET")
    sys.exit()
if POLY_IDX_END < 0:
    POLY_IDX_END = len(geometries) + POLY_IDX_START + 1

# Filter out any geometries with None as their category
if False:
    for i, category in enumerate(categories):
        if category is None:
            cat_none_idxs.append(i)
    geometries = [geometries[i] for i in range(len(geometries)) if i not in cat_none_idxs]
    categories = [categories[i] for i in range(len(categories)) if i not in cat_none_idxs]
    POLY_IDX_END = min(POLY_IDX_END, len(geometries)+1)

# Set up connection to the EO data service and specify the collection and bands to use
print("Setting up connection...")
if CONNECTION == 'des':
    connection = openeo.connect("https://openeo.digitalearth.se")
    print("EO service URL:", "https://openeo.digitalearth.se")
    if COLLECTION == 'SENTINEL2_L2A':
        collection = "s2_msi_l2a"
    bands = {"b01": 60, "b02": 10, "b03": 10, "b04": 10, "b05": 20, "b06": 20, "b07": 20, "b08": 10, "b8a": 20, "b09": 60, "b11": 20, "b12": 20}
    # Separate bands into bands_10x10, bands_20x20, bands_60x60, based on values in the bands dictionary
    bands_10x10 = [key for key, value in bands.items() if value == 10]
    bands_20x20 = [key for key, value in bands.items() if value == 20]
    bands_60x60 = [key for key, value in bands.items() if value == 60]
    band_types = {"10x10": bands_10x10, "20x20": bands_20x20, "60x60": bands_60x60}
elif CONNECTION == 'cop':
    connection = openeo.connect(url="openeo.dataspace.copernicus.eu")
    collection = COLLECTION
    if collection == 'SENTINEL1_GRD':
        bands = ['VV', 'VH']
        band_types = {"10x10": bands}
    elif collection == 'SENTINEL2_L2A':
        bands = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20}
        bands_10x10 = [key for key, value in bands.items() if value == 10]
        bands_20x20 = [key for key, value in bands.items() if value == 20]
        bands_60x60 = [key for key, value in bands.items() if value == 60]
        band_types = {"10x10": bands_10x10, "20x20": bands_20x20, "60x60": bands_60x60}
connection.authenticate_oidc()
print("Connection has been set up!")

# Begin processing the polygons
print("Processing polygons from the GPKG file...")
poly_idxs = np.arange(POLY_IDX_START, POLY_IDX_END)

outer_ctr = 0
with time_measurement("Putting jobs for all polygons"):  
    for poly_idx in poly_idxs:

        # Extract the current polygon, where first coordinate is longitude and second is latitude
        polygon = geometries[poly_idx]

        # Based on the above polygon, create the minimum bounding box containing the polygon
        # (but with some extra space around the polygon)
        min_lon = np.min(polygon[:, 0])
        max_lon = np.max(polygon[:, 0])
        min_lat = np.min(polygon[:, 1])
        max_lat = np.max(polygon[:, 1])

        # Need to ensure we get a square satellite image in the end
        # In the below, when negative, treated instead as relative margin on each side
        min_lat, min_lon, max_lat, max_lon, _, _ = create_square_bounding_box(
            min_lat, min_lon, max_lat, max_lon, width_km=MARGIN_PER_SIDE
        )

        # Get Sentinel 2 data for the bounding box
        with time_measurement(f" Processing polygon idx {poly_idx}/{POLY_IDX_END}..."):

            # Download the data
            for key, bands in band_types.items():
                job_designation = f"Area_{poly_idx+1}" + "_" + key

                # Base the filenames on the polygon number poly_idx and the key (10x10, 20x20, 60x60)
                file_name_nc = os.path.join(SAVE_PATH, f"Area_{poly_idx+1}" + "_" + key + '_image.nc')
                # Continue if it already exists.
                if os.path.exists(file_name_nc):
                    print(f"Skipping {file_name_nc} as it already exists.")
                    continue

                # Based on last year, specify date-span as the full season the next year
                mm_yy_start = '-'.join(DATE_SPAN[0].split('-')[1:])
                mm_yy_end = '-'.join(DATE_SPAN[1].split('-')[1:])
                new_start = str(timestamps[poly_idx] + 1) + '-' + mm_yy_start
                new_end = str(timestamps[poly_idx] + 1) + '-' + mm_yy_end
                next_year_date_span = [new_start, new_end]

                # Load the data cube
                cube = connection.load_collection(
                    collection,
                    spatial_extent={
                        "west": min_lon,
                        "south": min_lat,
                        "east": max_lon,
                        "north": max_lat,
                    },
                    temporal_extent=next_year_date_span,
                    bands=bands
                )

                job = cube.create_job(
                    out_format="netCDF",
                    options={"max_files": 200},
                    title=f"AILA-{job_designation}",
                    description=f"Downloads the bounding square of the given polygon nr {poly_idx}",
                )

                # Download the data
                try:
                    result = job.start_and_wait().download_results()
                except openeo.rest.JobFailedException as e:
                    # Find id of the job that failed based on e.
                    if False:
                        import re
                        import uuid
                        uuid_pattern = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE)
                        match = uuid_pattern.search(str(e))
                        found_uuid = uuid.UUID(match.group(0))
                        logs=connection.job(found_uuid).logs()
                        print(logs)
                        sys.exit()
                    else:
                        continue
            
                # Save the .nc file to disk
                nc_file_path = None
                for path, info in result.items():
                    if '.nc' in str(path):
                        nc_file_path = path
                        break
                if nc_file_path:
                    shutil.move(nc_file_path, file_name_nc)
                    print(f"Saved {file_name_nc}")
                else:
                    print("No .nc file found in the results.")
                    sys.exit()

            # Also save the associated polygon (in original coordinates) as an .npy file
            np.save(os.path.join(SAVE_PATH, f"Area_{poly_idx+1}" + '_polygon.npy'), polygon)
