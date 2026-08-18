import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import geopandas as gpd
import geopy.distance
import numpy as np
from other_scripts.utils import create_square_bounding_box, get_layer_data_from_gpkg
from functions.gdf_utils import get_polygon_gdf
import datetime
import fiona
import shapely as sh
import geopandas as gpd
from shapely.strtree import STRtree
import pandas as pd
import matplotlib.cm as cm
import math
from paths import GPKGS
from tests.gdf_tests import check_duplicate_values, check_duplicate_polygons

"""
Script for doing some checks and light filtering of a gdf file. Main feature is extracting all polygons and giving them individual rows for polygons
"""

#
date=datetime.datetime.now()
date=date.strftime("%Y-%m-%d_%H:%M")
save_dir = os.path.join("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/img_runs/gdpk_runs/",date)
os.makedirs(save_dir, exist_ok=True)


PATH_TO_GEORG_GPKG=GPKGS/"ApprovedPastures_2-7Years_startafter2017.gpkg"
VISUALIZE_RESULTS=True

#INITIAL ANALYSIS
gdf = gpd.read_file(PATH_TO_GEORG_GPKG)
gdf.head()
gdf.info()
print(gdf.crs)




#Creates new Gdf with one polygon per row, regardless of the geometry structure in the original gdf
gdf_polygons=get_polygon_gdf(gdf)
gdf_polygons.head()
gdf_polygons.info()
print(gdf_polygons.crs)




print(gdf_polygons.geometry.geom_type.value_counts())
print("Pre pruning", len(gdf_polygons))

#Calculating area of polygons and removing any with too small area
gdf_polygons["calc_area"] = gdf_polygons.geometry.area
gdf_polygons=gdf_polygons[gdf_polygons["calc_area"]>=100]


#Checking how many duplicate values exist for the different columns and prints it out
check_duplicate_values(gdf_polygons)

#Seeing if there are any identical polygons in the set
matches=check_duplicate_polygons(gdf_polygons)
print("Number of duplicate polygons", len(matches))

#If wanting to check something in original dataframe later
og_idx_df=gdf_polygons.copy()

#Resetting index
gdf_polygons=gdf_polygons.reset_index(drop=True)
print("Final number polygons: ", len(gdf_polygons))


out_path=GPKGS/"polygons_test.gpkg"
gdf_polygons.to_file(
    out_path,
    layer="polygons",
    driver="GPKG"
)

#Checking small areas
if VISUALIZE_RESULTS:
    small_areas=gdf_polygons[gdf_polygons["calc_area"]<=150]
    id_counts = gdf_polygons["id_orig"].value_counts()
    shared_ids = id_counts[id_counts > 1].index
    small_with_shared_id = small_areas[small_areas["id_orig"].isin(shared_ids)]
    for id_val, group in gdf_polygons.groupby("id_orig"):
        if id_val not in small_areas["id_orig"].values:
            continue

        fig, ax = plt.subplots()

        group[group["calc_area"]>=150].plot(ax=ax,color="blue")
        group[group["calc_area"]<=150].plot(ax=ax,color="red")
        plt.savefig(os.path.join(save_dir, f"Min_area{id_val}_with_larger.png"), bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close('all')