import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import geopandas as gpd
import geopy.distance
import numpy as np
from other_scripts.utils import create_square_bounding_box, get_layer_data_from_gpkg, read_image_and_polygon
import datetime
import fiona
import shapely as sh
import geopandas as gpd
from shapely.strtree import STRtree
import pandas as pd
import matplotlib.cm as cm
import math
from shapely.geometry import box
import json
from paths import GPKGS, NC


"""
Extracting all images from the NC files for each year and checking wether the polygons exist within each image, and saving the area name of the image in which each polygon is most aligned for every year.
"""

date=datetime.datetime.now()
date=date.strftime("%Y-%m-%d_%H:%M")
save_dir = os.path.join("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/img_runs/Aligning/",date)
os.makedirs(save_dir, exist_ok=True)

LOAD_PATH = NC
YEARS=["2018","2019","2020","2021","2022","2023","2024","2025","year-after-data"]
PATH_TO_GPKG=GPKGS/"filtered_polygons02-20.gpkg"
START_IDX=0
END_IDX=-1
IDXS_TO_USE = None

gdf = gpd.read_file(PATH_TO_GPKG)
gdf.head()
gdf.info()
print(gdf.crs)

gdf["Years_Areas"] = [[] for _ in range(len(gdf))] 
gdf.head()
gdf.info()
print(gdf.crs)

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

    area_names = np.full(len(gdf),None)  # For each polygon, find which area name --> min_dist to center
    min_dists = np.full(len(gdf),999999)
    for file_idx in range(START_IDX, end_idx):

        # Skip certain indices or activities
        if IDXS_TO_USE is not None and file_idx not in IDXS_TO_USE:
            continue

        im, poly, dates, y, x, spat_ref = read_image_and_polygon(files[file_idx],return_all=True)
        
        im_min_lon=np.min(x)
        im_min_lat=np.min(y)
        im_max_lon=np.max(x)
        im_max_lat=np.max(y)

        #Turn into shapely polygon to be able to use shapely methods :)
        im_bbox=[im_min_lon,im_min_lat,im_max_lon,im_max_lat]
        im_poly = box(*im_bbox)

        # Get the base filename
        base = os.path.basename(files[file_idx])    # 'Area_2_10x10_image.nc'
        # Remove the suffix after '_10x10' (or more generally after '_')
        area = base.split("_10x10")[0]  # 'Area_2'
        # Keeps the area name where distance between polygon center and patch center is the smallest
        # (i.e. the polygon is most centered in the image)
        for idx, row in gdf.iterrows(): # For every polygon in gdf
            if row.geometry.within(im_poly):    # Is poly in current image?
                gpkg_poly = row.geometry
                centroid = gpkg_poly.centroid
                center = im_poly.centroid
                dist = centroid.distance(center)
                # Update if distance is smaller than prev min dist
                if dist < min_dists[idx]:
                    min_dists[idx] = dist
                    area_names[idx] = area
    
    for idx, row in gdf.iterrows(): # For every polygon in gdf (this year)
        if area_names[idx] is not None:
            gdf.at[idx, "Years_Areas"].append(os.path.join(year,area_names[idx]))



mask = gdf["Years_Areas"].str.len() > 0
gdf_with_images = gdf[mask]
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])
print(gdf_with_images.sample(10)[["Years_Areas"]])


gdf["Years_Areas_json"] = gdf["Years_Areas"].apply(json.dumps)

gdf_to_save = gdf.drop(columns=["Years_Areas"])

out_path=GPKGS/"centered_filtered_polygons02-20.gpkg"
gdf_to_save.to_file(
    out_path,
    layer="polygons",
    driver="GPKG"
)