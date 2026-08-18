import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import numpy as np
from other_scripts.utils import read_image_and_polygon, align_poly_with_image, create_square_bounding_box, mlp_inference, get_layer_data_from_gpkg
from classes import MLP5
import geopy.distance
import skimage.draw as skdraw
import datetime
import geopandas as gpd




def extract_polys_from_directory(path_to_directory: str) -> list:
    """
    Reads files in a given directory and only extracts numpy arrays with their corresponding file names. Skips files and directories
    that aren't np.arrays
    
    :param path_to_directory: The directory path
    :type path_to_directory: str
    :return: Lists for the polygons (np.arrays) and their corresponding filenames
    :rtype: list
    """
    
    polys=[]
    file_names=[]

    for filename in os.listdir(path_to_directory):
        temp_path=os.path.join(path_to_directory,filename)

        #Want to skip other directories etc.
        if not os.path.isfile(temp_path):
            continue
        
        try:
            polys.append(np.load(temp_path,allow_pickle=False))
            file_names.append(filename)
        except Exception:
            #Not a np file
            continue
    
    #Might add hash things later to check if the arrays are unique. Maybe in a different function
    return polys, file_names


def poly_lon_lat_data(polygons: iter) -> np.array:
    """
    Extracts max-min coordinates from one or multiple polygons
    
    :param polygons: Collection of polygons
    :type polygons: iter
    :return: (4xN) array containing all min-max bounding boxes
    :rtype: np.array
    """

    min_lon=[]
    max_lon=[]
    min_lat=[]
    max_lat=[]

    for poly in polygons:
        min_lon.append(np.min(poly[:,0]))
        max_lon.append(np.max(poly[:,0]))
        min_lat.append(np.min(poly[:,1]))
        max_lat.append(np.max(poly[:,1]))

    return np.array([min_lon,max_lon,min_lat,max_lat])

def plot_polys_on_map(polygons,path_to_map,ax=None,year="",color=None):
    """
    Drawing polygons on map of sweden, if an ax is not provided a new one will be created
    
    :param polygons: Collection of polygons
    :param path_to_map: String of the path to the map
    :param ax: Ax
    :param year: Optional Year marker for plotting
    :param color: Color of the markers
    :return: The fig and ax where the map is drawn 
    """
    if ax is None:
        fig, ax=plt.subplots()

        sweden = gpd.read_file(path_to_map)

        # Filter to only Sweden
        sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']
        sweden.boundary.plot(ax=ax, color="black", linewidth=1)

        lon_lats=poly_lon_lat_data(polygons)

        # Scatter plot min-coords
        ax.scatter(lon_lats[0], lon_lats[2], s=0.5, label=f'Polygons {year} (n={len(polygons)})',color=color)
        ax.grid(True)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        return fig, ax
    
    else:
        sweden = gpd.read_file(path_to_map)

        # Filter to only Sweden
        sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']
        sweden.boundary.plot(ax=ax, color="black", linewidth=1)

        lon_lats=poly_lon_lat_data(polygons)

        # Scatter plot min-coords
        ax.scatter(lon_lats[0], lon_lats[2], s=0.5, label=f'Polygons {year} (n={len(polygons)})',color=color)
        ax.grid(True)
        ax.set_title(f'Polygons {year}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

def poly_overlap(polygons):
    """
    Returns indicies of pairs of overlapping polygon-bounding boxes from a collection of polygons. 
    
    :param polygons: Collection of polygons
    :return: Unique pairs of overlapping polygons as a np.array with shape (N polygons, 2)
    """
    lonlats=poly_lon_lat_data(polygons).T
    
    min_x = lonlats[:, 0][:, None]
    max_x = lonlats[:, 1][:, None]
    min_y = lonlats[:, 2][:, None]
    max_y = lonlats[:, 3][:, None]

    #Gives a NxN matrix of True where there is overlap between polygons with given indicies and False otherwise
    overlap = (
    (min_x < max_x.T) &
    (max_x > min_x.T) &
    (min_y < max_y.T) &
    (max_y > min_y.T)
    )

    np.fill_diagonal(overlap, False)

    i, j = np.where(overlap)
    pairs = np.column_stack((i, j))

    #gets rid of duplicates
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    return pairs

def overlap_same_year(pairs,years):
    """
    Given the same polygon-indicies are used in the years array and the pairs, returns onfly the pairs that overlap and share year
    
    :param pairs: Pair of polygons who's boundingboxes overlap
    :param years: Array of years for a given polygon index
    :return: The polygons overlapping within same years
    """
    #Want to check if there is overlap with any in the same year
    years=np.array(years)
    same_year_mask = years[pairs[:, 0]] == years[pairs[:, 1]]
    same_year_pairs = pairs[same_year_mask]
    return same_year_pairs

def complete_overlap(polygons,eps=0.001):
    """
    Returns indices of pairs of completely overlapping polygon-bounding boxes from a collection of polygons. 
    :param polygons: Collection of polygons
    :param eps: overlap margin (difference between polygons are smaller than eps)
    :return: Unique pairs of completely overlapping polygons as a np.array with shape (N polygons, 2)
    """
    lonlats=poly_lon_lat_data(polygons).T
    
    min_x = lonlats[:, 0][:, None]
    max_x = lonlats[:, 1][:, None]
    min_y = lonlats[:, 2][:, None]
    max_y = lonlats[:, 3][:, None]

    overlap = (
    (np.abs(min_x - min_x.T) < eps) &
    (np.abs(max_x - max_x.T) < eps) &
    (np.abs(min_y - min_y.T) < eps) &
    (np.abs(max_y - max_y.T) < eps)
    )
    #print("OVERLAP: ", overlap)
    np.fill_diagonal(overlap, False)

    i, j = np.where(overlap)
    pairs = np.column_stack((i, j))
    #gets rid of duplicates
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    return pairs

    
    
if __name__=="__main__":


    date=datetime.datetime.now()
    date=date.strftime("%Y-%m-%d_%H:%M")
    save_dir = os.path.join("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/img_runs/poly_runs/",date)
    os.makedirs(save_dir, exist_ok=True)

    #Works but plotting is ugly AF when doing multiple years. Will have to fix later
    LOAD_PATH = "../sen2a-data-mark-georg"
    YEARS=["2020","2021","2022","2023"]
    MAP_PATH="../country-borders/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    VISUALIZE=False
    #fig,ax=plt.subplots(1,len(YEARS))
    i=0
    
    cmap = plt.get_cmap("tab10")  # or "tab20", "viridis", ...
    colors = [cmap(i/len(YEARS)) for i in range(len(YEARS))]
    full_polys=[]
    full_filenames=[]
    full_years=[]


    for YEAR in YEARS:
        polys,file_names = extract_polys_from_directory(os.path.join(LOAD_PATH,YEAR))
        full_polys.extend(polys)
        full_filenames.extend(file_names)
        full_years.extend([YEAR]*len(polys))
        if VISUALIZE:
            plot_polys_on_map(polys,MAP_PATH,ax=ax[i],year=YEAR,color=colors[i])
        i+=1
    
    pairs=poly_overlap(full_polys)
    print("PAIRS: ", pairs)
    print("Overlapping Areas: ", full_filenames[pairs[0,0]]+" "+ full_years[pairs[0,0]],full_filenames[pairs[0,1]]+" "+ full_years[pairs[0,1]])
    print("No overlapping: ", len(pairs))
    print("Total: ", len(full_polys))

    same_year_pairs=overlap_same_year(pairs,full_years)
    print("Number of overlap within years: ", len(same_year_pairs) )