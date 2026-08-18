import sys, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import geopandas as gpd
import netCDF4
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms
import segmentation_models_pytorch as smp
import random
from shapely.geometry import Polygon, MultiPolygon
from torchmetrics.functional import jaccard_index as tm_jaccard
from torchmetrics.functional import f1_score as tm_f1
import scipy
from scipy.special import expit
import geopy.distance
from pyproj import Transformer
import fiona
import pickle
from typing import List, Sequence, Tuple
import sklearn.metrics
from contextlib import contextmanager
import skimage.draw as skdraw
from pathlib import Path
import rasterio
import glob
import cv2

@contextmanager
def time_measurement(msg):
    thrown_exception: Exception = None
    try:
        ts_start = datetime.now(timezone.utc)
        yield  # Execute the block of code inside the 'with' statement
    except Exception as e:
        thrown_exception = e
        raise
    finally:
        ts_end = datetime.now(timezone.utc)
        duration = (ts_end - ts_start).total_seconds()
        if thrown_exception:
            print(
                f"{msg} raised exception {type(thrown_exception)}"
                f" {thrown_exception}after {duration}s"
            )
        else:
            print(f"{msg} took {duration}s")

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

# Below function not used
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, do_show=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if do_show:
            plt.show()
        # If not show, then it will later be saved to a file instead

def align_poly_with_image(poly, im_height, im_width, margin=0.0, clip_if_poly_outside=True):
    # In the below im_width and im_height correspond to a slightly larger image than
    # the minimum bounding box of the polygon (because of the margin). Thus
    # we need to adjust the polygon by moving it slightly to the right and down,
    # and when scaling with the image width and height, we need to scale it down.
    
    # Based on min_lat, min_lon, max_lat, max_lon, calculate the distance in km
    # of the latitudal span and the longitudinal span
    min_lon = np.min(poly[:,0])
    max_lon = np.max(poly[:,0])
    min_lat = np.min(poly[:,1])
    max_lat = np.max(poly[:,1])
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).kilometers
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).kilometers
    if lat_span_km > lon_span_km:
        # latitude corresponds to the y-axis, longitude to the x-axis
        # thus in this case we are considering an image which is taller than it is wide
        im_width_tight = im_width / (1 + 2 * margin) * lon_span_km / lat_span_km
        im_height_tight = im_height / (1 + 2 * margin)
    else:
        im_height_tight = im_height / (1 + 2 * margin) * lat_span_km / lon_span_km
        im_width_tight = im_width / (1 + 2 * margin)

    # Begin aligning the polygon with the image
    poly_aligned = poly.copy()
    poly_aligned[:,0] = poly_aligned[:,0] - np.min(poly_aligned[:,0])
    poly_aligned[:,1] = poly_aligned[:,1] - np.min(poly_aligned[:,1])
    poly_aligned[:,0] = poly_aligned[:,0] / np.max(poly_aligned[:,0]) * im_width_tight  # poly[:,0] is the longitude (x-axis)
    poly_aligned[:,1] = poly_aligned[:,1] / np.max(poly_aligned[:,1]) * im_height_tight  # poly[:,1] is the latitude (y-axis)
    
    # Now move poly_aligned so that it is centered in the image
    poly_aligned[:,0] = poly_aligned[:,0] + (im_width - im_width_tight) / 2
    poly_aligned[:,1] = poly_aligned[:,1] + (im_height - im_height_tight) / 2
    
    # Assert some dimensions etc
    assert poly_aligned.shape[1] == 2
    if clip_if_poly_outside:
        poly_aligned[:,0] = np.clip(poly_aligned[:,0], 0, im_width)
        poly_aligned[:,1] = np.clip(poly_aligned[:,1], 0, im_height)

    # Return the aligned polygon
    return poly_aligned

def create_square_bounding_box(min_lat, min_lon, max_lat, max_lon, width_km=-0.25):
    """
    Creates a square bounding box around a center point.

    Args:
        min_lat (float): Smallest endpoint latitude.
        min_lon (float): Smallest endpoint longitude.
        max_lat (float): Largest endpoint latitude.
        max_lon (float): Largest endpoint longitude.
        width_km (float): Width of the square bounding box in kilometers.
                          If non-positive, then interpret it instead as a relative
                          margin on each side around the center point (adaptively set
                          based on original min_lat, min_lon, max_lat, max_lon).

    Returns:
        tuple: (min_lat, min_lon, max_lat, max_lon) representing the square bounding box.
    """

    if width_km <= 0:
        # Based on min_lat, min_lon, max_lat, max_lon, calculate the distance in km
        # of the latitudal span and the longitudinal span
        lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).kilometers
        lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).kilometers

        # Since we want a square bounding box, we need to set the width_km to the
        # maximum of the two spans, and also add the relative margin
        margin = -width_km * 2  # Times 2 since it is a margin on each side
        width_km = max(lat_span_km, lon_span_km)
        width_km += margin * width_km

    # Calculate the distance in kilometers for the given width
    half_width_km = width_km / 2

    # Calculate the coordinates of the corners of the square bounding box
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    center_point = geopy.Point(center_lat, center_lon)
    north_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 0)
    south_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 180)
    east_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 90)
    west_point = geopy.distance.distance(kilometers=half_width_km).destination(center_point, 270)

    # Extract latitude and longitude values for square bounding box
    min_lat = south_point.latitude
    min_lon = west_point.longitude
    max_lat = north_point.latitude
    max_lon = east_point.longitude

    # Calculate the distance in km of the latitudal span and the longitudinal span
    lat_span_km = geopy.distance.distance((min_lat, min_lon), (max_lat, min_lon)).kilometers
    lon_span_km = geopy.distance.distance((min_lat, min_lon), (min_lat, max_lon)).kilometers
    try:
        assert np.abs(lat_span_km - width_km) < 1e-2 and np.abs(lon_span_km - width_km) < 1e-2
    except:
        print(f"lat_span_km: {lat_span_km}, lon_span_km: {lon_span_km}, width_km: {width_km}")
        raise ValueError("The latitudal and longitudinal spans do not match the width_km")

    # Return the square bounding box
    return min_lat, min_lon, max_lat, max_lon, lat_span_km, lon_span_km

def read_image_and_polygon(filename_im, band_upsampling_approach='bilinear', return_poly=True):
    # Read image, polygon and dates of an area
    # For band_upsampling_approach, the possibilities are:
    # - 'nearest': Use cv2.resize with interpolation=cv2.INTER_NEAREST
    # - 'bilinear': Use cv2.resize with interpolation=cv2.INTER_LINEAR
    # - 'bicubic': Use cv2.resize with interpolation=cv2.INTER_CUBIC
    
    # Update filename_im to match the fact that there are 10x10, 20x20 and 60x60
    # counterparts of the same area, and that the filename is the same for all
    filenames_im = [filename_im, filename_im.replace('10x10', '20x20'), filename_im.replace('10x10', '60x60')]

    # Process similarly for the different band resolutions
    ims_all_bands = []
    for i, filename_im in enumerate(filenames_im):
        im_nc = netCDF4.Dataset(filename_im)
        if '10x10' in filename_im:
            x = im_nc.variables['x'][:]
            y = im_nc.variables['y'][:]

        # im_nc above is a netCDF4.Dataset object, now we want to extract the data from it
        # and convert it to a numpy array
        # Create an np array that has dimension H x W x C x T, where C is the number of bands,
        # and T is the number of timestamps. We want to do this by fetching all variables in
        # the .nc file that contain the word "B" or "b".
        bands = []
        for variable in im_nc.variables.keys():
            if 'B' in variable or 'b' in variable:
                bands.append(im_nc.variables[variable][:, :, :][np.newaxis])

        # Each of the C elements in bands have shape (1, T, H, W).
        # Now from this use np to create im that has shape (H, W, C, T).
        im = np.concatenate(bands, axis=0)
        im = np.transpose(im, (2, 3, 0, 1))
        
        # Append to ims_all_bands
        ims_all_bands.append(im)

    # Next, we want to upsample the 20x20 and 60x60 bands to 10x10,
    # based on band_upsampling_approach. Use cv2 for this.
    H_upsampled, W_upsampled, _, T = ims_all_bands[0].shape
    size_upsampled = max(H_upsampled, W_upsampled)  # We want to upsample to a square image
    for i in range(len(ims_all_bands)):
        # Use cv2 to resize the image to the size of the 10x10 image
        curr_img = ims_all_bands[i]
        H, W, C, T = curr_img.shape
        # curr_img has shape H x W x C x T; iterate over the last dimension
        curr_img_upsampled = np.zeros((size_upsampled, size_upsampled, C, T))
        for t in range(T):
            if band_upsampling_approach == 'nearest':
                curr_img_upsampled[:, :, :, t] = cv2.resize(curr_img[:, :, :, t], (size_upsampled, size_upsampled), interpolation=cv2.INTER_NEAREST)
            elif band_upsampling_approach == 'bilinear':
                curr_img_upsampled[:, :, :, t] = cv2.resize(curr_img[:, :, :, t], (size_upsampled, size_upsampled), interpolation=cv2.INTER_LINEAR)
            elif band_upsampling_approach == 'bicubic':
                curr_img_upsampled[:, :, :, t] = cv2.resize(curr_img[:, :, :, t], (size_upsampled, size_upsampled), interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError(f"Unknown band_upsampling_approach: {band_upsampling_approach}")
        ims_all_bands[i] = curr_img_upsampled

    # We now want to create im, which contains ALL bands (that currently reside in ims_all_bands)
    # and the order of the bands in im should match the variable bands above.
    ims_10x10 = ims_all_bands[0]
    ims_20x20 = ims_all_bands[1]
    ims_60x60 = ims_all_bands[2]
    bands = {"b01": 60, "b02": 10, "b03": 10, "b04": 10, "b05": 20, "b06": 20, "b07": 20, "b08": 10, "b8a": 20, "b09": 60, "b11": 20, "b12": 20}
    bands_10x10 = [key for key, value in bands.items() if value == 10]
    bands_20x20 = [key for key, value in bands.items() if value == 20]
    bands_60x60 = [key for key, value in bands.items() if value == 60]
    im = []
    for band_name, _ in bands.items():
        if band_name in bands_10x10:
            im.append(ims_10x10[:, :, bands_10x10.index(band_name), :][:, :, np.newaxis, :])
        elif band_name in bands_20x20:
            im.append(ims_20x20[:, :, bands_20x20.index(band_name), :][:, :, np.newaxis, :])
        elif band_name in bands_60x60:
            im.append(ims_60x60[:, :, bands_60x60.index(band_name), :][:, :, np.newaxis, :])
        else:
            raise ValueError(f"Unknown band_name: {band_name}")
    im = np.concatenate(im, axis=2)

    # Load the polygon
    filename_im = filenames_im[0]
    if return_poly:
        poly = np.load(filename_im.replace('_image.nc', '_polygon.npy').replace('10x10_', ''))
    else:
        poly = None

    # Get the timestamps from im_nc in date format i.e. YYYY-MM-DD
    dates = netCDF4.num2date(im_nc.variables['t'], im_nc.variables['t'].units)
    dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Return the image, polygon, the dates, and the image y- and x-spans
    return im, poly, dates, y, x

def _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz):
	if post_filt_sz == 1:
		return pred_map_binary_list, pred_map_binary_thin_list
	H, W = pred_map.shape
	for list_idx, pred_map_binary in enumerate(pred_map_binary_list):
		tmp_map = np.zeros_like(pred_map)
		tmp_map_thin = np.zeros_like(pred_map)
		count_map = np.zeros_like(pred_map)
		for i_start in range(post_filt_sz):
			for j_start in range(post_filt_sz):
				for i in range(i_start, H // post_filt_sz):
					for j in range(j_start, W // post_filt_sz):
						count_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						curr_patch = pred_map_binary[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						curr_patch_thin = pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)][i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz]
						if np.count_nonzero(curr_patch) >= np.prod(curr_patch.shape) // 2:
							tmp_map[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
						if np.count_nonzero(curr_patch_thin) >= np.prod(curr_patch_thin.shape) // 2:
							tmp_map_thin[i * post_filt_sz : (i + 1) * post_filt_sz, j * post_filt_sz : (j + 1) * post_filt_sz] += 1
		tmp_map[count_map == 0] = 0
		count_map[count_map == 0] = 1
		tmp_map /= count_map
		assert np.min(tmp_map) >= 0 and np.max(tmp_map) <= 1
		pred_map_binary = tmp_map >= 0.50
		pred_map_binary_list[list_idx] = pred_map_binary

		tmp_map_thin[count_map == 0] = 0
		tmp_map_thin /= count_map
		assert np.min(tmp_map_thin) >= 0 and np.max(tmp_map_thin) <= 1
		pred_map_binary_thin = tmp_map_thin >= 0.50
		pred_map_binary_thin_list[min(list_idx, len(thresh_thin_cloud) - 1)] = pred_map_binary_thin

		# 'Aliasing effect' after this filtering can cause BOTH thin and regular cloud to be active at the same time -- give prevalence to regular
		pred_map_binary_thin_list[0][pred_map_binary_list[0]] = 0

	return pred_map_binary_list, pred_map_binary_thin_list

# Setup MLP-computation function
def mlp_inference(img, means, stds, models, batch_size, thresh_cloud, thresh_thin_cloud, post_filt_sz, device='cpu', predict_also_cloud_binary=False):
	H, W, input_dim = img.shape
	img_torch = torch.reshape((torch.Tensor(img).to(device) - means) / stds, [H * W, input_dim])
	pred_map_tot = 0.0
	pred_map_binary_tot = 0.0
	for model in models:
		pred_map = np.zeros(H * W)
		pred_map_binary = np.zeros(H * W)
		for i in range(0, H * W, batch_size):
			curr_pred = model(img_torch[i : i + batch_size, :])
			pred_map[i : i + batch_size] = curr_pred[:, 0].cpu().detach().numpy()
			if predict_also_cloud_binary:
				pred_map_binary[i : i + batch_size] = curr_pred[:, 1].cpu().detach().numpy()
		pred_map = np.reshape(pred_map, [H, W])
		if predict_also_cloud_binary:
			pred_map_binary = np.reshape(expit(pred_map_binary), [H, W]) >= 0.5
		else:
			pred_map_binary = np.zeros_like(pred_map)#pred_map >= thresh_cloud[-1] <<--- overwritten anyway
			
		# Average model predictions
		pred_map_tot += pred_map / len(models)
		pred_map_binary_tot += pred_map_binary.astype(float) / len(models)
		
	# Return final predictions
	pred_map = pred_map_tot
	if predict_also_cloud_binary:
		pred_map_binary = pred_map_binary_tot >= 0.5
	else:
		pred_map_binary_list = []
		pred_map_binary_thin_list = []
		for thresh in thresh_cloud:
			pred_map_binary_list.append(pred_map_tot >= thresh)
		for thresh in thresh_thin_cloud:
			# Below: A thin cloud is a thin cloud only if it is above the thin thresh AND below the regular cloud thresh
			pred_map_binary_thin_list.append(np.logical_and(pred_map_tot >= thresh, pred_map_tot < thresh_cloud[0]))

	# Potentially do post-processing on the cloud/not cloud (binary)
	# prediction, so that it becomes more spatially coherent
	pred_map_binary_list, pred_map_binary_thin_list = _mlp_post_filter(pred_map_binary_list, pred_map_binary_thin_list, pred_map, thresh_thin_cloud, post_filt_sz)

	# Return
	return pred_map, pred_map_binary_list, pred_map_binary_thin_list

def get_layer_data_from_gpkg(path_to_gpkg, keep_only_unique_ids=False):
    # List all layers in the GeoPackage using fiona
    layers = fiona.listlayers(path_to_gpkg)

    # Initialize dictionaries to store data for each layer
    layer_data = {}

    # Setup a transformer to convert coordinates from EPSG:4326 to EPSG:3857
    transformer = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)

    # Read each layer and store its data
    category_exists = False
    for layer in layers:
        gdf = gpd.read_file(path_to_gpkg, layer=layer)
        polys = []  # Will store np-arrays of polygons
        geom_types = []
        ids = []
        uuids = []
        block_types = []
        timestamps = []
        categories = []
        agoslags = []
        for idx, row in gdf.iterrows():
            poly_data = row['geometry']
            geom_types.append(poly_data.geom_type)
            lon_lats = []
            for point in poly_data.exterior.coords:
                # Due to always_xy=True, the order of output is (lon, lat)
                # according to the documentation of pyproj
                lon, lat = transformer.transform(point[0], point[1])
                lon_lats.append((lon, lat))
            poly_np = np.array(lon_lats)
            polys.append(poly_np)
            ids.append(row['id'])
            uuids.append(row['uuid'])
            block_types.append(row['typ'])
            agoslags.append(row['agoslag'])
            timestamps.append(row['redigerat_datum'])
            if 'kategori' in row:
                categories.append(row['kategori'])
                category_exists = True
            else:
                categories.append(None)
        # Store data in the dictionary
        layer_data[layer] = {
            "polys": polys,
            "geom_types": geom_types,
            "ids": ids,
            "uuids": uuids,
            "block_types": block_types,
            "timestamps": timestamps,
            "category": categories,
            "agoslags": agoslags
        }

    # Filter out duplicate ids in each layer
    if keep_only_unique_ids:
        for layer, data in layer_data.items():
            unique_ids = set(data["ids"])
            unique_polys = []
            unique_geom_types = []
            unique_uuids = []
            unique_block_types = []
            unique_timestamps = []
            unique_categories = []
            unique_agoslags = []

            for idx, id_ in enumerate(data["ids"]):
                if id_ in unique_ids:
                    unique_polys.append(data["polys"][idx])
                    unique_geom_types.append(data["geom_types"][idx])
                    unique_uuids.append(data["uuids"][idx])
                    unique_block_types.append(data["block_types"][idx])
                    unique_timestamps.append(data["timestamps"][idx])
                    unique_categories.append(data["category"][idx])
                    unique_agoslags.append(data["agoslags"][idx])
                    unique_ids.remove(id_)
            # Update the layer data with unique values
            layer_data[layer] = {
                "polys": unique_polys,
                "geom_types": unique_geom_types,
                "ids": list(set(data["ids"])),  # Convert to set to remove duplicates
                "uuids": unique_uuids,
                "block_types": unique_block_types,
                "timestamps": unique_timestamps,
                "category": unique_categories,
                "agoslags": unique_agoslags
            }

    return layer_data, category_exists