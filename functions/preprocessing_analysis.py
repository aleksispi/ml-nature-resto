import numpy as np
from paths import GPKGS, ZARR, ROOT, CLOUD_MASKS
from classes import zarr_class
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from functions.utils import make_savedir
import os
import datetime
from pathlib import Path
import functions.plotting_functions as pf
import pickle
import geopandas as gpd
import classes.zarr_class as zc
from matplotlib.ticker import MaxNLocator, FuncFormatter


def plot_outliers(cube,poly_idxs = None,savefolder="linnea_testar_grejer_runs",additional_info="Outlier images"):
    """
    Plots an image grid of some outliers, taken from the mean std plot. These specifically are images where the image mean in band
    2-5 have a mean <0 (see mean std plot). 
    """
    savedir = make_savedir(savefolder,additional_info=additional_info) 
    if poly_idxs is None:
        poly_idxs = range(cube.p_dim)

    outlier_images = []
    times = []
    titles = [] 

    for poly_idx in poly_idxs:
        ts = cube.cloud_free_time(poly_idx)
        for t in ts:
            img = cube.get_unpadded_img(poly_idx,t).astype(np.float64)
            img_mean = np.mean(img,axis=(1,2))
            # img = np.asarray(cube.reflectance.oindex[poly_idx, t], dtype=float)     # Doing this instead to see if the results are the same
            # img[img==-9999] = np.nan
            # img_mean = np.nanmean(img,axis=(1,2))
            
            if np.any(img_mean[1:4] < 0):
                # Plot grid
                padded_img = np.asarray(cube.reflectance.oindex[poly_idx, t], dtype=float)
                outlier_images.append(padded_img)
                times.append(t)
                titles.append(np.round(np.min(img_mean[1:4])))

    outlier_images = np.stack(outlier_images,axis=0)
    print("Shape of outlier_images: ", np.shape(outlier_images))

    # Plot image grid
    num_images = np.shape(outlier_images)[0]
    ncols = np.ceil(np.sqrt(num_images)).astype(int)
    nrows = np.ceil(num_images / ncols).astype(int)
    fig = plt.figure(figsize=(32, 36))
    
    for i, t in enumerate(times):
        ax = fig.add_subplot(nrows,ncols,i+1)
        year=cube.timestamps[t].year
        fig, ax = pf.plot_rgb(outlier_images[i,:,:,:],year,normalized=True,fig=fig,ax=ax,title=titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.suptitle(additional_info + f", {num_images} images", fontsize=36)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(savedir/f"outlier_image_grid.png")
    plt.close()

def analyzing_outlier_blob(filename, cube):
    """
    Analysing images in blob from mean std plot (mean <0 for some bands).
    """
    neg_img_mask = np.load("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/linnea_experimentation/neg_img_mask.npy", allow_pickle=True)      
    #Loads dictionary of p ts outlier pairs
    with open(filename, "rb") as f:
        outliers = pickle.load(f)

    gdf=cube.get_gdf()
    ps=np.array([int(p) for p in outliers.keys()])
    outliers_in_neg_img_mask = 0
    tot_outliers=0
    for p in ps:
        ts=[int(t) for t in outliers[p]]
        # imgs=cube.get_normalized_images(p,ts)
        outliers_in_neg_img_mask+=np.sum(neg_img_mask[p,ts])
        tot_outliers+=len(ts)
    
    print("Number of outlier blob images in neg img mask:", outliers_in_neg_img_mask)
    print("Total images in outlier blob: ", tot_outliers)
    
      
            
def plot_imgs_with_neg_values(cube, savedir):
    """
    Plots and image grid of all images that contain at least one negative pixel.
    """
    poly_idxs = range(cube.p_dim)
    negative_imgs = []
    subfig_titles = []

    for poly_idx in poly_idxs:
        ts = cube.cloud_free_time(poly_idx)
        # ts = cube.valid_time(poly_idx)
        for t in ts:
            img = cube.get_unpadded_img(poly_idx,t).astype(np.float64)
            if np.any(img < 0): 
                padded_img = np.asarray(cube.reflectance.oindex[poly_idx, t], dtype=float)
                negative_imgs.append(padded_img)
                # subfig_titles.append(f"({poly_idx},{t})")  
                subfig_titles.append(np.min(img))

    negative_imgs = np.stack(negative_imgs, axis=0)   
    fig = pf.plot_image_grid(negative_imgs,subfig_titles)
    fig.savefig(savedir/"negative_imgs_grid")        
    plt.close()

def make_neg_img_mask(cube):
    """
    Finds all images that contain negative values, and mark them as true. 
    """
    neg_img_mask = np.zeros((cube.p_dim,cube.t_dim),dtype=bool)
    
    poly_idxs = range(cube.p_dim)

    for poly_idx in poly_idxs:
        ts = cube.valid_time(poly_idx)
        for t in ts:
            img = cube.get_unpadded_img(poly_idx,t).astype(np.float64)
            if np.any(img < 0): 
                neg_img_mask[poly_idx,t] = True
  
    np.save(ROOT/"linnea_experimentation/neg_img_mask.npy", neg_img_mask)


def plot_number_of_clouds():
    # Biased cloud mask
    cube=zc.Zarr(ZARR,GPKGS/'final02-20.gpkg')
    # cube=zc.Zarr(ZARR,GPKGS/'final02-20.gpkg',cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_15_15_10.npy")
    im_ids = cube.get_im_id()
    im_ids = im_ids!=-1
    cloud_mask = cube.get_cloud_mask()
    # kept = np.where(cube.get_cloud_mask() and im_ids)
   
    frac_kept_biased = []
    for year in range(2018,2026):
        time_ids = cube.range_to_indices(str(year)+"-01-01", str(year)+"-12-31")
        frac_kept = np.round(np.sum(cloud_mask[:,time_ids])/np.sum(im_ids[:,time_ids]),decimals=2)
        frac_kept_biased.append(frac_kept)

    # Unbiased cloud mask
    cube=zc.Zarr(ZARR,GPKGS/'final02-20.gpkg',cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    cloud_mask = cube.get_cloud_mask()
    print("Total images: ", np.sum(im_ids))
    print("Kept after cloud filter: ", np.sum(cloud_mask))
    print("Fraction: ", np.sum(cloud_mask)/np.sum(im_ids))
    exit()
    frac_kept_unbiased = []
    for year in range(2018,2026):
        time_ids = cube.range_to_indices(str(year)+"-01-01", str(year)+"-12-31")
        frac_kept = np.round(np.sum(cloud_mask[:,time_ids])/np.sum(im_ids[:,time_ids]),decimals=2)
        frac_kept_unbiased.append(frac_kept)

    savedir = make_savedir("linnea_testar_grejer_runs",additional_info="FONTSIZE Cloud filter comparison new thresh") 
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    years = range(2018,2026)
    # Plot lines
    ax.plot(years, 100*np.array(frac_kept_unbiased), marker='o', linewidth=2, label='Without offset')
    ax.plot(years, 100*np.array(frac_kept_biased), marker='o', linewidth=2, label='With -1000 offset')

    # Labels and title
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Images kept(%)", fontsize=16)
    ax.set_title("Cloud filter comparison", fontsize=18)

    # Axis formatting
    ax.set_xticks(years)
    ax.set_ylim(0, 100)

    # Grid
    ax.tick_params(axis='both', labelsize=16) 
    ax.grid(True, linestyle='--', alpha=0.3)

    # Legend
    ax.legend(frameon=False,fontsize=16)

    # Improve layout
    plt.tight_layout()
 
    # Save
    plt.savefig(savedir/"comparison_plot.png", dpi=300)

def plot_cluster_sizes():
    """
    How many polygon clusters are there of each size?
    """
    cube=zc.Zarr(ZARR,GPKGS/'final02-20.gpkg',cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    # Cluster histogram (polygons per cluster)
    savedir = make_savedir("linnea_testar_grejer_runs",additional_info=f"Cluster histogram") 
    gdf_with_clusters = gpd.read_file(GPKGS/"polygons_with_clusters_dist_1500.gpkg")  
    cluster_sizes = gdf_with_clusters['cluster'].value_counts().sort_index()    # Counts polygons per cluster
        
    plt.figure(figsize=(8, 5))
    bins = np.arange(cluster_sizes.min(), cluster_sizes.max() + 2) - 0.5
    plt.hist(cluster_sizes, bins=bins, edgecolor='black')

    plt.xlabel("Cluster size (# polygons)",fontsize=16)
    plt.ylabel("Number of clusters",fontsize=16)
    plt.title("Distribution of Cluster Sizes",fontsize=18)
    plt.tick_params(axis='both', labelsize=16) 
    plt.xticks(np.arange(cluster_sizes.min(), cluster_sizes.max() + 1)) # Place ticks at integers
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savedir/"cluster_histogram.png")

def plot_cloudy_and_non_cloudy(cube,poly_id,filename,savedir,title):
    """
    Plot cloudy and non cloudy in separate image grids
    """
    ts = cube.cloud_free_time(poly_id)
    cube.plot_image_grid(poly_id,ts,savedir,filename=filename+"_non_cloudy",title = title+f" {len(ts)} images.",plot_poly=True)

    ts = cube.valid_time(poly_id)[~np.isin(cube.valid_time(poly_id), ts)]
    cube.plot_image_grid(poly_id,ts,savedir,filename=filename+"_cloudy",title = title+f" {len(ts)} images.",plot_poly=True,select_cloudy=True)
    plt.close()    

if __name__=="__main__": 
    path_to_gpkg = GPKGS/'final02-20.gpkg'
    cube = zarr_class.Zarr(ZARR,path_to_gpkg)

    # OUTLIER ANALYSIS ---------------------------------------------------------------------------------------------------------------------
    # plot_outliers(cube,savefolder="linnea_testar_grejer_runs",additional_info="Outlier images, unpadded (using mean < 0)")
    # analyzing_outlier_blob("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/img_runs/data_statistics/outliers/outliers.pkl",cube)   # Result: only 38 out of 418 images from outlier blob is also a negative value img
    # savedir = make_savedir("linnea_testar_grejer_runs",additional_info="Imgs with negative values") 
    # plot_imgs_with_neg_values(cube, savedir)
    # make_neg_img_mask(cube)
    # neg_img_mask = np.load("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/linnea_experimentation/neg_img_mask.npy")
    # print("Times for negative images:", np.where(np.any(neg_img_mask, axis=0))[0])
    # print("Number of times tot: ", len(np.where(np.any(neg_img_mask, axis=0))[0]))
    # print("Polygons that have negative images:", np.where(np.any(neg_img_mask, axis=1))[0])
    # print("Number of polygons tot: ", len(np.where(np.any(neg_img_mask, axis=1))[0]))

    # CLOUD FILTER TEST ---------------------------------------------------------------------------------------------------------------------
    # savedir = make_savedir("cloud_filter_runs",additional_info='Comparing old and new mask - random polygon') 
    # # poly_id = 118
    # poly_id = np.random.randint(cube.p_dim)
    # plot_cloudy_and_non_cloudy(cube,poly_id,filename=f"Polygon_{poly_id}_old_mask", savedir=savedir, 
    #                                         title= f"Polygon {poly_id}, old mask.")
    # # Update mask
    # cube.load_cloud_mask(CLOUD_MASKS/"cloud_mask_04_17_thresholds_20_15_10.npy")
    # plot_cloudy_and_non_cloudy(cube,poly_id,filename=f"Polygon_{poly_id}_new_mask", savedir=savedir, 
    #                                         title= f"Polygon {poly_id}, new mask.")