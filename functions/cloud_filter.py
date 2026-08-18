import os, sys
import numpy as np
from other_scripts.utils import mlp_inference
import torch
import skimage.draw as skdraw 
from other_scripts.classes import MLP5
from paths import COT_MODEL
import matplotlib.pyplot as plt
from functions.utils import make_savedir
from functions.plotting_functions import plot_image_grid


def cloud_filter(im_series,poly_masks,im_ids,THRESHOLD_THICKNESS_IS_CLOUD=0.015,THRESHOLD_THICKNESS_IS_THIN_CLOUD=0.015,POLY_CLOUD_FRAC_THRESH=0.01,MLP_POST_FILTER_SZ = 1): 
    """
    Filters out cloudy pictures in im_series, and returns a clean im_series

    Args:
        im_series: ndarray of shape (T, C, H, W) 
        poly_masks: polygon masks that are aligned with img in image coordinates.
        im_ids: image ids that are -1 if the image does not exists (padding on time axis).
        THRESHOLD_THICKNESS_IS_CLOUD (float): if COT predicted above this, then predicted as 'opaque cloud' ("thick" cloud)
        THRESHOLD_THICKNESS_IS_THIN_CLOUD (float): if COT predicted above this, then predicted as 'thin cloud' <-- set to the same as the 
                                                opaque cloud threshold by default, i.e. it becomes a binary task (cloudy / clear) instead
        POLY_CLOUD_FRAC_THRESH (float): If fraction of cloudy pixels within polygon is below this, then consider as cloud-free
        MLP_POST_FILTER_SZ (int):  1 --> no filtering, >= 2 --> majority vote within that-sized square
    Returns:
        im_series with the cloudy images removed.
    """
    
    DEVICE = 'cpu'
    # The below model weights can be directly downloaded from this link:
    # https://drive.google.com/drive/mobile/folders/14xTbLHPxaPznemG7ShE0DMC9zJsNU_hr?usp=sharing
    # (See also the COT model repo here: https://github.com/aleksispi/ml-cloud-opt-thick)
    COT_MODEL_LOAD_PATH = [COT_MODEL/'2023-08-10_11-49-01/model_it_2000000', COT_MODEL/'2023-08-10_11-49-22/model_it_2000000', COT_MODEL/'2023-08-10_11-49-49/model_it_2000000',
                        COT_MODEL/'2023-08-10_11-50-44/model_it_2000000', COT_MODEL/'2023-08-10_11-51-11/model_it_2000000', COT_MODEL/'2023-08-10_11-51-36/model_it_2000000',
                        COT_MODEL/'2023-08-10_11-51-49/model_it_2000000', COT_MODEL/'2023-08-10_11-52-02/model_it_2000000', COT_MODEL/'2023-08-10_11-52-24/model_it_2000000',
                        COT_MODEL/'2023-08-10_11-52-47/model_it_2000000']
    # Make some things into lists
    if not isinstance(COT_MODEL_LOAD_PATH, list):
        MODEL_LOAD_PATH = [COT_MODEL_LOAD_PATH]
    if not isinstance(THRESHOLD_THICKNESS_IS_CLOUD, list):          
        THRESHOLD_THICKNESS_IS_CLOUD = [THRESHOLD_THICKNESS_IS_CLOUD]
    if not isinstance(THRESHOLD_THICKNESS_IS_THIN_CLOUD, list):
        THRESHOLD_THICKNESS_IS_THIN_CLOUD = [THRESHOLD_THICKNESS_IS_THIN_CLOUD]

    # Setup and load COT prediction model (Pirinen et al. 2024)
    models = []
    for model_load_path in COT_MODEL_LOAD_PATH:
        model = MLP5(11, 1, apply_relu=True)
        model.load_state_dict(torch.load(model_load_path, map_location=DEVICE)) 
        model.to(DEVICE)
        models.append(model)

    # Load means and stds based on the synthetic data
    means = torch.Tensor(np.array([0.64984976, 0.4967399, 0.47297233, 0.56489476, 0.52922534, 0.65842892, 0.93619591, 0.90525398, 0.99455938, 0.45607598, 0.07375734, 0.53310641, 0.43227456])).to(DEVICE)
    stds = torch.Tensor(np.array([0.3596485, 0.28320853, 0.27819884, 0.28527526, 0.31613214, 0.28244289, 0.32065759, 0.33095272, 0.36282185, 0.29398295, 0.11411958, 0.41964159, 0.33375454])).to(DEVICE)
    SKIP_BAND_10 = True                                                                                                                                                                                                    
    if SKIP_BAND_10:
        means = means[[0,1,2,3,4,5,6,7,8,9,11,12]]
        stds = stds[[0,1,2,3,4,5,6,7,8,9,11,12]]
        # Note: in im_series, band 10 is already not there
        # since it is not present in the Sentinel 2 L2A data
    means = means[1:]  # Also skip band 1 as not used for COT model
    stds = stds[1:]

    T, C, H, W = im_series.shape
    cloudy_flag=np.ones(T,dtype=bool) 

    pred_maps_cot = []          # For plotting
    cot_titles = []
    pred_maps_binary = []       # For plotting
    binary_titles = []

    # Iterate over the timeseries
    for t in range(T):
        if im_ids[t] == -1:
            cloudy_flag[t] = False
            continue
        
        # Perform prediction
        img = im_series[t,:, :, :]

        img = img[1:, :, :]  # Skip band 1 for COT model

        img = img.transpose(1,2,0)
       
        pred_map, pred_map_binary_list, pred_map_binary_list_thin = mlp_inference(img, means, stds, models, H*W, THRESHOLD_THICKNESS_IS_CLOUD,
                                                                                THRESHOLD_THICKNESS_IS_THIN_CLOUD, MLP_POST_FILTER_SZ, DEVICE)
        pred_map_binary = pred_map_binary_list[0]
        pred_map_binary_thin = pred_map_binary_list_thin[0]
        
        # Check which pixels within the polygon are deemed as cloudy
   
        pred_map_binary_poly = pred_map_binary[poly_masks[t,:,:]]
        frac_binary_poly = np.sum(pred_map_binary_poly) / len(pred_map_binary_poly)
        pred_cloudy_poly = frac_binary_poly > POLY_CLOUD_FRAC_THRESH

        if pred_cloudy_poly:
            cloudy_flag[t] = False

    # Diagnostics -----------------------------------------------------------------------------------------------------------
    #     # Print fraction of pixels deemed as cloudy, both in total and within the polygon
    #     frac_binary = np.sum(pred_map_binary) / (H*W)
    #     pred_cloudy = frac_binary > 0
    #     # print("Fraction of pixels deemed as cloudy (total, polygon):", frac_binary*100, frac_binary_poly*100) 

    #     # COT   
    #     cot_titles.append((np.round(np.nanmin(pred_map),decimals=3), np.round(np.nanmax(pred_map),decimals=3)))
    #     pred_map[np.isnan(pred_map)] = 0
    #     pred_maps_cot.append(pred_map)

    #     # BINARY
    #     pred_maps_binary.append(0.0 + 2*pred_map_binary + pred_map_binary_thin)
    #     if pred_cloudy:
    #         binary_titles.append((np.round(100*frac_binary,decimals=2), np.round(100*frac_binary_poly,decimals=2)))   # %tot vs %poly
    #     else:
    #         binary_titles.append((np.round(100*frac_binary,decimals=2)))


    #  PLOT IMAGE GRIDS OF PRED MAPS  

    # savedir = make_savedir("linnea_testar_grejer_runs", additional_info = "Cloud pred maps (different thin and thick thresholds)")
    # thresh = THRESHOLD_THICKNESS_IS_CLOUD[0]
    # poly_frac_thresh = POLY_CLOUD_FRAC_THRESH
    
    # labels = ["Clear" if flag else "Cloudy" for flag in cloudy_flag[im_ids!=-1]]
    # fig = plot_image_grid(im_series[im_ids!=-1], subfig_titles = labels, 
    #                       fig_title = f"Im_series. Thickness threshold {thresh}, poly frac thresh {poly_frac_thresh}. {T} images.")
    # fig.savefig(savedir/f"Im_series_image_grid_thresholds_{int(thresh*1000)}_and_{int(poly_frac_thresh*1000)}")
    
    # fig = plot_pred_map_grid(pred_maps_cot, subfig_titles = cot_titles, 
    #                          fig_title = f"COT map. Thickness threshold {thresh}, poly frac thresh {poly_frac_thresh}. {T} images.")
    # fig.savefig(savedir/f"COT_image_grid_thresholds_{int(thresh*1000)}_and_{int(poly_frac_thresh*1000)}")

    # fig = plot_pred_map_grid(pred_maps_binary, subfig_titles = binary_titles, 
    #                          fig_title = f"Binary cloud map. Thickness threshold {thresh}, poly frac thresh {poly_frac_thresh}. {T} images.")
    # fig.savefig(savedir/"Binary_image_grid")

    return im_series[cloudy_flag,:,:,:], cloudy_flag

def plot_pred_map_grid(pred_maps, subfig_titles = [], fig_title = None, figsize = (32, 32), title_space = 0.04):
    """
    Plots an pred map grid of the specified pred maps.  

    Args:
        pred_maps: list of all pred maps, length N. 
        subfig_titles: list of titles for the individual images. Length N. 
        fig_title: title of the figure.
        figsize: figure size.
        title_space: vertical space for the figure title.

    Returns:
        fig: figure that contains the plot.
    """
    # Calculates rows and columns for a "square" grid
    N = len(pred_maps)
    ncols = np.ceil(np.sqrt(N)).astype(int)
    nrows = np.ceil(N / ncols).astype(int)
    fig = plt.figure(figsize=figsize)

    # Fills out with empty titles if there aren't enough
    if len(subfig_titles) < N:
        n_missing = N - len(subfig_titles)
        subfig_titles.extend([' '] * n_missing)
    
    for i in range(N):
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.imshow(pred_maps[i], vmin=0, vmax=1, cmap='gray')
        ax.set_title(subfig_titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if fig_title is None:
        fig_title = f"Image grid of {N} pred maps"

    fig.suptitle(fig_title, fontsize=36)
    plt.tight_layout(rect=[0, 0, 1, 1-title_space])

    return fig
