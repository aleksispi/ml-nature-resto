import os, sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw
import matplotlib.dates as mdates


def align_poly(poly, im_height, im_width, x, y, clip_if_poly_outside=True):
    """
    Aligns a polygon to and image using lon lat coordinates. 

    Args:
        poly (ndarray): polygon coordinates with longitude in the first column and latitude in the second, shape (N,2)
        im_height (int): image height
        im_width (int): image width
        x: longitude coordinate axis
        y: latitude coordinate axis
        clip_if_poly_outside: clips poly if outside image.

    Returns:
        poly_aligned: a polygon in image coordinates that is aligned to the specified image. 
    """
    # Look at satellite img:
    xx, yy = np.meshgrid(x, y)

    # transformer = pyproj.Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    # lon, lat = transformer.transform(xx, yy) 
    lon, lat = xx,yy    # if in the same coord system, this should work

    sat_min_lon=np.min(lon)
    sat_max_lon=np.max(lon) 
    sat_min_lat=np.min(lat)
    sat_max_lat=np.max(lat)
    
    scale_lon = (im_width-1)/(sat_max_lon-sat_min_lon)
    scale_lat = (im_height-1)/(sat_max_lat-sat_min_lat)     # Changed

    poly_aligned = poly.copy()
    poly_aligned[:,0] = (poly[:,0]-sat_min_lon)*scale_lon
    poly_aligned[:,1] =(sat_max_lat-poly[:,1])*scale_lat
   
    # Assert some dimensions etc
    assert poly_aligned.shape[1] == 2
    if clip_if_poly_outside:
        poly_aligned[:,0] = np.clip(poly_aligned[:,0], 0, im_width)
        poly_aligned[:,1] = np.clip(poly_aligned[:,1], 0, im_height)

    poly_mask = mask_poly(poly_aligned,im_height,im_width)

    
    if poly_mask.sum() == 0:
        return None, None

    # Return the aligned polygon
    return poly_aligned, poly_mask

def get_world_axis(x0,y0,im_height,im_width,meter_per_pxl=10):
    """
    Gets the full x and y axis from corner coordinate (x0,y0)
    """
    x = x0 + np.arange(im_width)*meter_per_pxl
    y = y0 - np.arange(im_height)*meter_per_pxl
    return x,y

def mask_poly(poly_aligned, H, W):
    """
    Creates a mask that is True inside the polygon, and False outside. 
    """
    poly_aligned_int = np.floor(poly_aligned).astype(int)
    poly_mask = np.zeros((H, W), dtype=bool)
    rr, cc = skdraw.polygon(poly_aligned_int[:,1], poly_aligned_int[:,0])
    rr = np.clip(rr, 0, H-1)
    cc = np.clip(cc, 0, W-1)
    poly_mask[rr, cc] = True
    return poly_mask

def remove_img_padding(img):
    """
    Removes the -9999 padding of an image of shape (C,H,W). Returns the cropped image.
    """
    valid_mask = np.any(img != -9999, axis=0)  # shape (100, 100)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return img[:, row_min:row_max+1, col_min:col_max+1]


def plot_rgb(img,poly_aligned=None,fig=None,ax=None,title='Untitled'):
    """
    Extracts and plots the RGB image from a satellite image with 12 channels.

    Args:
        img: ndarray of shape (C,H,W) 
        poly_aligned: polygon that is aligned with img in image coordinates. Shape (N,2)
        fig, ax: to plot in. Will be supplied if unspecified.
        title: figure title

    Returns:
        fig, ax that contains the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Extract and show RGB image
    rgb_sub = img[[3, 2, 1], :, :]
    max_val = np.nanmax(rgb_sub)

    if max_val == 0 or np.isnan(max_val):
        rgb_img = np.zeros_like(rgb_sub)
    else:
        rgb_img = rgb_sub / max_val
    rgb_img = rgb_img.transpose(1, 2, 0)

    ax.imshow(rgb_img)
    if poly_aligned is not None:
        ax.plot(poly_aligned[:,0], poly_aligned[:,1], color='red', linewidth=2)
    
    ax.set_title(title, fontsize=16)

    return fig,ax


def make_hist(
    data,
    bins=30,
    figsize=(8, 5),
    title=None,
    xlabel="Value",
    ylabel="Frequency",
    color="C0",
    edgecolor="black",
    alpha=0.7,
    density=False,
    log=False,
    grid=True,
    grid_style="--",
    grid_alpha=0.4,
    tight_layout=True,
    **kwargs, 
):
    """General-purpose histogram generator."""

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        data,
        bins=bins,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        density=density,
        log=log,
        **kwargs,                # forward all additional parameters
    )

    # Labels & title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Grid
    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    if tight_layout:
        fig.tight_layout()

    return fig, ax


def plot_1dtime_series(data,timestamps,y_label="Data",title="1d Plot"):
    """Takes data and timestamps and creates a nice 1d plot of timeseries
    """

    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(timestamps, data, color="green",marker=".",linestyle='none')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()

    return fig,ax

def plot_image_grid(images, subfig_titles = [], fig_title = None, figsize = (32, 32), title_space = 0.04):
    """
    Plots an RGB image grid of the specified images. 

    Args:
        images: ndarray of shape (N,C,H,W), where N is the number of images. 
        subfig_titles: list of titles for the individual images. Length N. 
        fig_title: title of the figure.
        figsize: figure size.
        title_space: vertical space for the figure title.

    Returns:
        fig: figure that contains the plot.
    """
    # Calculates rows and columns for a "square" grid
    N = np.shape(images)[0]
    ncols = np.ceil(np.sqrt(N)).astype(int)
    nrows = np.ceil(N / ncols).astype(int)
    fig = plt.figure(figsize=figsize)

    # Fills out with empty titles if there aren't enough
    if len(subfig_titles) < N:
        n_missing = N - len(subfig_titles)
        subfig_titles.extend([' '] * n_missing)
    
    for i in range(N):
        ax = fig.add_subplot(nrows,ncols,i+1)
        fig, ax = plot_rgb(images[i,:,:,:],fig=fig,ax=ax,title=subfig_titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if fig_title is None:
        fig_title = f"Image grid of {N} images"

    fig.suptitle(fig_title, fontsize=36)
    plt.tight_layout(rect=[0, 0, 1, 1-title_space])

    return fig

if __name__=="__main__": 
    from paths import GPKGS, ZARR
    from classes import zarr_class
    from functions.utils import make_savedir
    path_to_gpkg = GPKGS/'final02-20.gpkg'
    cube = zarr_class.Zarr(ZARR,path_to_gpkg)
    # TRY OUT PLOTTING
    poly_id = 118
    ts = cube.cloud_free_time(poly_id)
    savedir = make_savedir("linnea_testar_grejer_runs", additional_info='Testing plot_image_grid in pf') 
    fig = plot_image_grid(cube.get_normalized_images(poly_id,ts))
    fig.savefig(savedir/"pf_image_grid")
    
    cube.plot_image_grid(poly_id,ts,savedir,filename="cube_image_grid",plot_poly=False)
        




