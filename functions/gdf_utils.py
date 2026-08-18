import geopandas as gpd
import geopy.distance
import numpy as np
import shapely as sh
import geopandas as gpd
from shapely.strtree import STRtree
import pandas as pd
import functions.utils as ut
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def extract_polygons(geom):
    """
    Recursively extract all Polygon objects from a geometry.
    Returns a list of Polygons.
    """
    polygons = []

    if geom.geom_type == "Polygon":
        polygons.append(geom)

    elif geom.geom_type == "MultiPolygon":
        polygons.extend(list(geom.geoms))

    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            polygons.extend(extract_polygons(g))

    # ignore other geometry types (LineString, Point, etc.)
    return polygons

def get_polygon_gdf(gdf):
    """
    Iterates through the rows, getting rid of all geometries that aren't polygons, multipolygons or geometrycollections.
    For multipolygons and geometry collections, extracts each individual polygon to a separate row
    
    Returns: Filtered gdf, list of the indices in the original gdf that contained multiple polygons
    """

    # List to store all polygons with attributes
    rows = []
    multi_polys=[]
    #Original file contains lines, points geometry collections etc. Extracting everything that is a polygon
    for idx, row in gdf.iterrows():
        polygons = extract_polygons(row.geometry)
        if len(polygons)>1:
            multi_polys.append(idx)
        for poly in polygons:
            new_row = row.copy()
            new_row.geometry = poly
            rows.append(new_row)

    # Create new flattened GeoDataFrame that only contains polygons
    gdf_polygons = gpd.GeoDataFrame(rows, columns=gdf.columns,crs=gdf.crs )
    return gdf_polygons, multi_polys

def plot_row(name,row,save_folder,fig=None,ax=None):
        """
        Method for plotting and saving a row of a gdf

        returns fig, ax for which the geometry was plotted on 
        """
        save_dir=ut.make_save_dir(save_folder,additional_info='')
        fig , ax= plt.subplots()
        row.plot(ax=ax,color="blue")
        plt.savefig(os.path.join(save_dir, f"{name}"), bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close('all')
        return fig, ax    

def plot_multi_geom_row(name,row,save_folder,fig=None,ax=None):
    """
    For rows containing multiple geometries, plots them in different colors
    in the same plot
    
    returns fig, ax for which the geometries are plotted on
    """
    save_dir=ut.make_save_dir(save_folder,additional_info='')
    fig , ax= plt.subplots()
    
    # Explode geometry into individual components (works for MultiPolygon & GeometryCollection)
    parts = gpd.GeoSeries(row.geometry).explode(index_parts=False)

    # Create a colormap with as many unique colors as needed
    colors = list(mcolors.TABLEAU_COLORS.values())  # 10 distinct colors
    # If more parts than colors, repeat colors
    colors = (colors * ((len(parts) // len(colors)) + 1))[:len(parts)]

    
    for i, geom in enumerate(parts):
        gpd.GeoSeries([geom]).plot(ax=ax, color=colors[i], label=f"Part {i+1}")

    plt.savefig(os.path.join(save_dir, f"{name}"), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close('all')
    return fig, ax  

